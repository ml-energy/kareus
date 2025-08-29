"""
Compare the accuracy of cost estimator
"""

import os
import time
import torch
import json
from cfuser.scheduler.perf_model import CostEstimator
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


def run_runner(
    rank,
    world_size,
    master_port,
    batch_size,
    height,
    width,
    num_inference_steps,
    attn_u_degree,
    attn_r_degree,
    non_attn_sp_degree,
    return_dict,
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
                ulysses_degree=non_attn_sp_degree,
                ring_degree=1,
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

    runner = CServeRunner(pretrained_model_name_or_path=engine_config.model_config.model, engine_config=engine_config)

    # warmup
    for i in range(4):
        req = ScheduledRequest(
            req_ids=[i],
            attn_ranks=list(range(non_attn_sp_degree)),
            non_attn_ranks=list(range(non_attn_sp_degree)),
            attn_ulysses_degree=non_attn_sp_degree,
            attn_ring_degree=1,
            non_attn_sp_degree=non_attn_sp_degree,
            input_config=input_config,
        )

        reqs = ScheduledRequests(requests=[req])
        runner.generate(reqs)

    # benchmark
    time_start = time.perf_counter()
    for i in range(4, 8):
        req = ScheduledRequest(
            req_ids=[i],
            attn_ranks=list(range(attn_u_degree * attn_r_degree)),
            non_attn_ranks=list(range(non_attn_sp_degree)),
            attn_ulysses_degree=attn_u_degree,
            attn_ring_degree=attn_r_degree,
            non_attn_sp_degree=non_attn_sp_degree,
            input_config=input_config,
        )
        reqs = ScheduledRequests(requests=[req])
        runner.generate(reqs)
    time_end = time.perf_counter()
    # print(f"time: {time_end - time_start}s")
    avg_time = (time_end - time_start) / 4

    # Store the avg_time in the shared dictionary
    if rank == 0:  # Only store from rank 0 to avoid conflicts
        return_dict["avg_time"] = avg_time

    get_runtime_state().destory_distributed_env()


def run_benchmark(
    world_size,
    master_port,
    batch_size,
    height,
    width,
    num_inference_steps,
    attn_u_degree,
    attn_r_degree,
    non_attn_sp_degree,
):
    # Create a shared manager and dictionary
    manager = torch.multiprocessing.Manager()
    return_dict = manager.dict()

    try:
        torch.multiprocessing.spawn(
            run_runner,
            args=(
                world_size,
                master_port,
                batch_size,
                height,
                width,
                num_inference_steps,
                attn_u_degree,
                attn_r_degree,
                non_attn_sp_degree,
                return_dict,
            ),
            nprocs=world_size,
            join=True,
        )
    except Exception as e:
        print(f"error: {e}")
        return 0.0

    return return_dict.get("avg_time", 0.0)


"""
python benchmark/component_scaling_efficiency/compare.py 2>&1 | tee log/benchmark/component_scaling_efficiency/compare.log
"""

if __name__ == "__main__":
    from cfuser.core.utils.zmq_utils import find_free_port

    ring_attn_scaling_efficiency_path = (
        "log/benchmark/component_scaling_efficiency/ring_scaling_efficiency/ring_scaling_efficiency.json"
    )
    non_attn_scaling_efficiency_path = (
        "log/benchmark/component_scaling_efficiency/non_attn_efficiency/non_attn_efficiency.json"
    )
    cost_estimator = CostEstimator(
        torch.cuda.device_count(), ring_attn_scaling_efficiency_path, non_attn_scaling_efficiency_path
    )

    output_json_path = "log/benchmark/component_scaling_efficiency/compare.json"

    if os.path.exists(output_json_path):
        with open(output_json_path, "r") as f:
            output_json = json.load(f)
    else:
        output_json = []

    hc = 24
    hs = 128
    for seq_len in [1024, 2048, 4096, 8192, 16384, 32768, 65536]:
        for bs in [1, 2, 4, 8, 16, 32]:
            height = 512
            width = seq_len * 16 * 16 // height

            steps = 10

            skip = False
            for item in output_json:
                if item["bs"] == bs and item["seq_len"] == seq_len and item["hc"] == hc and item["hs"] == hs:
                    print(f"skip {bs}, {seq_len}, {hc}, {hs}")
                    skip = True
                    break

            if skip:
                continue
            else:
                print(f"run {bs}, {seq_len}, {hc}, {hs}")

            try:
                naive_estimated_time, naive_u_degree, naive_r_degree = cost_estimator.naive_cost(bs, seq_len, hc, hs)
            except Exception as e:
                print(f"error: {e}")
                continue

            if naive_estimated_time == float("inf"):
                continue
            naive_estimated_time = naive_estimated_time * steps
            naive_time = run_benchmark(
                world_size=naive_u_degree * naive_r_degree,
                master_port=find_free_port(),
                batch_size=bs,
                height=height,
                width=width,
                num_inference_steps=steps,
                attn_u_degree=naive_u_degree,
                attn_r_degree=naive_r_degree,
                non_attn_sp_degree=naive_u_degree * naive_r_degree,
            )
            print(f"bs: {bs}, seq_len: {seq_len}, hc: {hc}, hs: {hs}")
            print(
                f"naive_estimated_time: {naive_estimated_time}, naive_u_degree: {naive_u_degree}, naive_r_degree: {naive_r_degree}"
            )
            print(f"naive average time: {naive_time}s")

            scaling_estimated_time, scaling_u_degree, scaling_r_degree = cost_estimator.scaling_efficiency(
                bs, seq_len, hc, hs
            )

            scaling_time = run_benchmark(
                world_size=scaling_u_degree * scaling_r_degree,
                master_port=find_free_port(),
                batch_size=bs,
                height=height,
                width=width,
                num_inference_steps=steps,
                attn_u_degree=scaling_u_degree,
                attn_r_degree=scaling_r_degree,
                non_attn_sp_degree=scaling_u_degree * scaling_r_degree,
            )
            scaling_estimated_time = scaling_estimated_time * steps
            print(
                f"scaling_estimated_time: {scaling_estimated_time}, scaling_u_degree: {scaling_u_degree}, scaling_r_degree: {scaling_r_degree}"
            )
            print(f"scaling average time: {scaling_time}s")

            disaggregated_estimated_time, mlp_min_gpus, attn_u_degree, attn_r_degree = (
                cost_estimator.disaggregated_scaling(bs, seq_len, hc, hs)
            )
            disaggregated_time = run_benchmark(
                world_size=mlp_min_gpus,
                master_port=find_free_port(),
                batch_size=bs,
                height=height,
                width=width,
                num_inference_steps=steps,
                attn_u_degree=attn_u_degree,
                attn_r_degree=attn_r_degree,
                non_attn_sp_degree=mlp_min_gpus,
            )
            disaggregated_estimated_time = disaggregated_estimated_time * steps
            print(
                f"disaggregated_estimated_time: {disaggregated_estimated_time}, mlp_min_gpus: {mlp_min_gpus}, attn_u_degree: {attn_u_degree}, attn_r_degree: {attn_r_degree}"
            )
            print(f"disaggregated average time: {disaggregated_time}s")

            output_json.append(
                {
                    "bs": bs,
                    "seq_len": seq_len,
                    "hc": hc,
                    "hs": hs,
                    "naive": {
                        "u_degree": naive_u_degree,
                        "r_degree": naive_r_degree,
                        "estimated_time": naive_estimated_time,
                        "real_time": naive_time,
                    },
                    "scaling": {
                        "u_degree": scaling_u_degree,
                        "r_degree": scaling_r_degree,
                        "estimated_time": scaling_estimated_time,
                        "real_time": scaling_time,
                    },
                    "disaggregated": {
                        "u_degree": attn_u_degree,
                        "r_degree": attn_r_degree,
                        "mlp_min_gpus": mlp_min_gpus,
                        "estimated_time": disaggregated_estimated_time,
                        "real_time": disaggregated_time,
                    },
                }
            )

            with open(output_json_path, "w") as f:
                json.dump(output_json, f, indent=2)
