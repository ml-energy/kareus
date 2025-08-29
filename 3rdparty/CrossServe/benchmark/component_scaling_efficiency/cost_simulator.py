import json
import time
import torch
from cfuser.scheduler.perf_model import CostEstimator

"""
compute pipeline:
comp_prologue->comm->comp_attn->comm->comp_epilogue
"""

LOG_DIR = "log_A100x4_80GB"

if __name__ == "__main__":
    ring_attn_scaling_efficiency_path = (
        f"{LOG_DIR}/benchmark/component_scaling_efficiency/ring_scaling_efficiency/ring_scaling_efficiency.json"
    )
    non_attn_scaling_efficiency_path = (
        f"{LOG_DIR}/benchmark/component_scaling_efficiency/non_attn_efficiency/non_attn_efficiency.json"
    )
    cost_estimator = CostEstimator(
        torch.cuda.device_count(), ring_attn_scaling_efficiency_path, non_attn_scaling_efficiency_path
    )

    output_json_path = f"{LOG_DIR}/benchmark/component_scaling_efficiency/cost_simulator.json"
    output_json = []

    hc = 24
    hs = 128
    steps = 50
    threshold = float("inf")  # 2.5 minutes
    # strategy = "economy"
    strategy = "fastest"

    start_time = time.perf_counter()
    # for bs in [2]:
    #     for seq_len in [4096]:
    for bs in [1, 2, 4, 8, 16, 32]:
        for seq_len in [1024, 2048, 4096, 8192, 16384, 32768, 65536]:

            naive_cost, naive_u_degree, naive_r_degree = cost_estimator.naive_cost(bs, seq_len, hc, hs)
            if naive_cost == float("inf"):
                continue
            print(f"bs: {bs}, seq_len: {seq_len}, hc: {hc}, hs: {hs}")
            print(
                f"naive_cost: {naive_cost * steps} s, naive_u_degree: {naive_u_degree}, naive_r_degree: {naive_r_degree}"
            )

            scaling_cost, scaling_u_degree, scaling_r_degree = cost_estimator.scaling_efficiency(
                bs, seq_len, hc, hs, threshold=threshold / steps, strategy=strategy
            )
            print(
                f"scaling_cost: {scaling_cost * steps} s, scaling_u_degree: {scaling_u_degree}, scaling_r_degree: {scaling_r_degree}"
            )

            cost, mlp_min_gpus, attn_u_degree, attn_r_degree = cost_estimator.disaggregated_scaling(
                bs, seq_len, hc, hs, threshold=threshold / steps, strategy=strategy
            )
            print(
                f"disaggregated_cost: {cost * steps} s, mlp_min_gpus: {mlp_min_gpus}, attn_u_degree: {attn_u_degree}, attn_r_degree: {attn_r_degree}"
            )
            print("-" * 100)

            output_json.append(
                {
                    "bs": bs,
                    "seq_len": seq_len,
                    "hc": hc,
                    "hs": hs,
                    "naive_cost": naive_cost,
                    "naive_u_degree": naive_u_degree,
                    "naive_r_degree": naive_r_degree,
                    "scaling_cost": scaling_cost,
                    "scaling_u_degree": scaling_u_degree,
                    "scaling_r_degree": scaling_r_degree,
                    "disaggregated_cost": cost,
                    "mlp_min_gpus": mlp_min_gpus,
                    "attn_u_degree": attn_u_degree,
                    "attn_r_degree": attn_r_degree,
                }
            )

    with open(output_json_path, "w") as f:
        json.dump(output_json, f, indent=2)

    end_time = time.perf_counter()
    print(f"Total scheduling time: {end_time - start_time} s")
