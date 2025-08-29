import pandas as pd
from dataclasses import dataclass
import json
import math

from cfuser.logger import init_logger

logger = init_logger(__name__)


@dataclass
class ScalingConfig:
    ulysses_degree: int
    ring_degree: int
    scaling_factor: float
    actual_speedup: float
    batch_size: int
    seq_len: int
    estimated_time: float


class CostEstimator:
    """ """

    def __init__(
        self,
        num_gpus: int,
        ring_attn_scaling_efficiency_path: str,
        non_attn_scaling_efficiency_path: str,
        # comm_scaling_efficiency_path: str, # TODO: add comm scaling-efficiency later, is it necessary? anyway we'll overlap them then
    ):

        self.num_gpus = num_gpus
        self.num_layers = 19
        self.num_single_layers = 38

        try:
            with open(ring_attn_scaling_efficiency_path, "r") as f:
                self.ring_attn_scaling_efficiency = json.load(f)
        except FileNotFoundError:
            logger.warning(f"File not found: {ring_attn_scaling_efficiency_path}")
            logger.warning(f"run benchmark/component_scaling_efficiency to get the cost_estimator json file")

        try:
            with open(non_attn_scaling_efficiency_path, "r") as f:
                self.non_attn_scaling_efficiency = json.load(f)
        except FileNotFoundError:
            logger.warning(f"File not found: {non_attn_scaling_efficiency_path}")
            logger.warning(f"run benchmark/component_scaling_efficiency to get the cost_estimator json file")

        self.ring_attn_dict = {}
        self.non_attn_dict = {}

        for data in self.ring_attn_scaling_efficiency:
            key = (
                data["bs"],
                data["seq_len"],
                data["hc"],
                data["hs"],
                data["ulysses_world_size"],
                data["ring_attn_world_size"],
            )
            self.ring_attn_dict[key] = data["avg_time"]

        for data in self.non_attn_scaling_efficiency:
            key = (data["bs"], data["seq_len"], data["hc"], data["hs"], data["ulysses_world_size"])
            self.non_attn_dict[key] = (
                data["avg_time_multimodal_prologue"],
                data["avg_time_unimodal_prologue"],
                data["avg_time_multimodal_epilogue"],
                data["avg_time_unimodal_epilogue"],
            )

    def sleep(self, seconds: int):
        import time

        time.sleep(seconds / 1000)

    def compute_cost(
        self,
    ):
        """
        Compute the cost of the given trace
        """
        pass

    def naive_cost(self, bs, seq_len, hc, hs, gpu_num=None):
        """
        Serving every Request in all GPUs
        """
        if gpu_num is None:
            gpu_num = self.num_gpus

        multimodal_prologue, unimodal_prologue, multimodal_epilogue, unimodal_epilogue = self.non_attn_dict.get(
            (bs, seq_len, hc, hs, gpu_num), (float("inf"), float("inf"), float("inf"), float("inf"))
        )

        min_cost = float("inf")
        u_degree = 0
        r_degree = 0
        for i in range(int(math.log2(gpu_num)) + 1):
            attn_cost = self.ring_attn_dict.get((bs, seq_len, hc, hs, 2**i, gpu_num // (2**i)), float("inf"))
            if attn_cost < min_cost:
                min_cost = attn_cost
                u_degree = 2**i
                r_degree = gpu_num // (2**i)

        estimate_cost = (multimodal_prologue + min_cost + multimodal_epilogue) * self.num_layers + (
            unimodal_prologue + min_cost + unimodal_epilogue
        ) * self.num_single_layers

        return estimate_cost, u_degree, r_degree

    def scaling_efficiency(self, bs, seq_len, hc, hs, strategy="fastest", max_gpus=None, threshold=None):
        """
        Serving every Request by the whole scaling-efficiency
        """

        assert strategy in ["fastest", "economy"]

        if max_gpus is None:
            max_gpus = self.num_gpus

        if strategy == "fastest":
            min_cost = float("inf")
            min_u_degree = 0
            min_r_degree = 0

            for parallel_degree in [1, 2, 4, 8]:
                if parallel_degree > max_gpus:
                    continue
                cost, u_degree, r_degree = self.naive_cost(bs, seq_len, hc, hs, parallel_degree)
                if cost < min_cost:
                    min_cost = cost
                    min_u_degree = u_degree
                    min_r_degree = r_degree

        elif strategy == "economy":

            if threshold is None:
                threshold = float("inf")

            base_cost, base_u_degree, base_r_degree = self.naive_cost(bs, seq_len, hc, hs, 1)

            min_cost = base_cost
            max_scaling_factor = 0
            min_u_degree = base_u_degree
            min_r_degree = base_r_degree

            for parallel_degree in [2, 4, 8]:
                if parallel_degree > max_gpus:
                    continue

                cost, u_degree, r_degree = self.naive_cost(bs, seq_len, hc, hs, parallel_degree)
                scaling_factor = (base_cost / cost) / parallel_degree
                if scaling_factor > 1:
                    continue

                if cost > threshold:
                    continue

                if scaling_factor > max_scaling_factor:
                    max_scaling_factor = scaling_factor
                    min_cost = cost
                    min_u_degree = u_degree
                    min_r_degree = r_degree

            if max_scaling_factor == 0:
                # it means no valid config under given threshold found, return the fastest config
                return self.scaling_efficiency(bs, seq_len, hc, hs, strategy="fastest", max_gpus=max_gpus)

        return min_cost, min_u_degree, min_r_degree

    def disaggregated_scaling(self, bs, seq_len, hc, hs, strategy="fastest", threshold=None, max_gpus=None):
        """
        Serving every Request by the disaggregated every component scaling-efficiency
        """

        assert strategy in ["fastest", "economy"]

        if strategy == "fastest":
            multimodal_prologue_min, unimodal_prologue_min, multimodal_epilogue_min, unimodal_epilogue_min = (
                float("inf"),
                float("inf"),
                float("inf"),
                float("inf"),
            )
            mlp_min_gpus = 0

            for i in range(int(math.log2(self.num_gpus)) + 1):
                multimodal_prologue, unimodal_prologue, multimodal_epilogue, unimodal_epilogue = self.non_attn_dict.get(
                    (bs, seq_len, hc, hs, 2**i), (float("inf"), float("inf"), float("inf"), float("inf"))
                )
                if (multimodal_prologue + multimodal_epilogue) * self.num_layers + (
                    unimodal_prologue + unimodal_epilogue
                ) * self.num_single_layers < (multimodal_prologue_min + multimodal_epilogue_min) * self.num_layers + (
                    unimodal_prologue_min + unimodal_epilogue_min
                ) * self.num_single_layers:
                    multimodal_prologue_min = multimodal_prologue
                    unimodal_prologue_min = unimodal_prologue
                    multimodal_epilogue_min = multimodal_epilogue
                    unimodal_epilogue_min = unimodal_epilogue
                    mlp_min_gpus = 2**i

            min_cost = float("inf")
            attn_u_degree = 0
            attn_r_degree = 0
            for u in range(int(math.log2(self.num_gpus)) + 1):
                for r in range(int(math.log2(self.num_gpus)) + 1):
                    attn_cost = self.ring_attn_dict.get((bs, seq_len, hc, hs, 1, 2**r), float("inf"))
                    if attn_cost < min_cost:
                        min_cost = attn_cost
                        attn_u_degree = 1
                        attn_r_degree = 2**r

            estimate_cost = (multimodal_prologue_min + min_cost + multimodal_epilogue_min) * self.num_layers + (
                unimodal_prologue_min + min_cost + unimodal_epilogue_min
            ) * self.num_single_layers

        elif strategy == "economy":
            if threshold is None:
                threshold = float("inf")

            if max_gpus is None:
                max_gpus = self.num_gpus

            # for attn, we need to find if parallel_degree=1 is the fastest config, if so we use it, if not  we use the scaling-best config
            attn_min_cost = float("inf")
            attn_u_degree = 0
            attn_r_degree = 0
            for u in range(int(math.log2(max_gpus)) + 1):
                for r in range(int(math.log2(max_gpus)) + 1):
                    attn_cost = self.ring_attn_dict.get((bs, seq_len, hc, hs, 1, 2**r), float("inf"))
                    if attn_cost < attn_min_cost:
                        attn_min_cost = attn_cost
                        attn_u_degree = 1
                        attn_r_degree = 2**r

            if attn_u_degree != 1 or attn_r_degree != 1:
                max_scaling_factor = 0.6  # 0.6 is the minimum scaling factor for attn we could tolerate
                base_attn_cost = self.ring_attn_dict.get((bs, seq_len, hc, hs, 1, 1), None)
                if base_attn_cost is not None:
                    for u in range(int(math.log2(max_gpus)) + 1):
                        for r in range(int(math.log2(max_gpus)) + 1):
                            if u != 0:
                                # TODO add ulysses scaling-efficiency later
                                continue
                            if 2**u == 1 and 2**r == 1:
                                continue
                            attn_cost = self.ring_attn_dict.get((bs, seq_len, hc, hs, 2**u, 2**r), float("inf"))
                            scaling_factor = (base_attn_cost / attn_cost) / (2**r * 2**u)
                            if scaling_factor > max_scaling_factor:
                                max_scaling_factor = scaling_factor
                                attn_min_cost = attn_cost
                                attn_u_degree = 2**u
                                attn_r_degree = 2**r

            assert attn_min_cost < float("inf")

            # for non-attn, we need to find the scaling-efficiency config under given threshold
            multimodal_prologue, unimodal_prologue, multimodal_epilogue, unimodal_epilogue = self.non_attn_dict.get(
                (bs, seq_len, hc, hs, 1), (float("inf"), float("inf"), float("inf"), float("inf"))
            )
            base_non_attn_cost = (multimodal_prologue + multimodal_epilogue) * self.num_layers + (
                unimodal_prologue + unimodal_epilogue
            ) * self.num_single_layers

            mlp_min_gpus = 1
            mlp_min_cost = base_non_attn_cost
            max_scaling_factor = 0

            for i in range(1, int(math.log2(max_gpus)) + 1):
                multimodal_prologue, unimodal_prologue, multimodal_epilogue, unimodal_epilogue = self.non_attn_dict.get(
                    (bs, seq_len, hc, hs, 2**i), (float("inf"), float("inf"), float("inf"), float("inf"))
                )

                mlp_cost = (multimodal_prologue + multimodal_epilogue) * self.num_layers + (
                    unimodal_prologue + unimodal_epilogue
                ) * self.num_single_layers

                scaling_factor = (base_non_attn_cost / mlp_cost) / (2**i)

                if scaling_factor > 1:
                    continue

                estimate_cost = mlp_cost + attn_min_cost * (self.num_layers + self.num_single_layers)
                if estimate_cost > threshold:
                    continue

                if scaling_factor > max_scaling_factor:
                    max_scaling_factor = scaling_factor
                    mlp_min_cost = mlp_cost
                    mlp_min_gpus = 2**i

            if max_scaling_factor == 0:
                # it means no valid config under given threshold found, return the fastest config
                return self.disaggregated_scaling(
                    bs, seq_len, hc, hs, strategy="fastest", threshold=threshold, max_gpus=max_gpus
                )

            estimate_cost = mlp_min_cost + attn_min_cost * (self.num_layers + self.num_single_layers)
        else:
            raise ValueError(f"Invalid strategy: {strategy}")

        return estimate_cost, mlp_min_gpus, attn_u_degree, attn_r_degree


class AttnPerfModel:
    def __init__(self, data_path: str):
        with open(data_path, "r") as f:
            self.ulysses_scaling_efficiency = json.load(f)
            self.seq_len_list = [item["seq_len"] for item in self.ulysses_scaling_efficiency]
            self.attn_scaling_efficiency = {}
            for item in self.ulysses_scaling_efficiency:
                if item["seq_len"] not in self.attn_scaling_efficiency:
                    self.attn_scaling_efficiency[item["seq_len"]] = []
                self.attn_scaling_efficiency[item["seq_len"]].append(item)

    def find_fastest_degree(self, seq_len: int, batch_size: int):
        if seq_len not in self.attn_scaling_efficiency:
            # find the closest by seq_len
            min_diff = float("inf")
            closest_len = None
            for refer_len in self.seq_len_list:
                abs_diff = abs(refer_len - seq_len)
                if abs_diff < min_diff:
                    min_diff = abs_diff
                    closest_len = refer_len
            seq_len = closest_len

        fastest_time = float("inf")
        fastest_degree = None
        for item in self.attn_scaling_efficiency[seq_len]:
            if item["bs"] == batch_size:
                if item["avg_time"] < fastest_time:
                    fastest_time = item["avg_time"]
                    fastest_degree = item["world_size"]
                else:
                    continue
            else:
                continue

        if fastest_degree is None:
            raise ValueError(f"No fastest degree found for seq_len {seq_len} and batch_size {batch_size}")

        return fastest_degree


class PerfModel:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = self.analyze_scaling_efficiency(pd.read_csv(data_path))

    def analyze_scaling_efficiency(self, df):
        # Group by all parameters except ulysses_degree, ring_attention_degree and Total Duration
        group_cols = [
            "model",
            "device",
            "if_torch_compile",
            "batch_size",
            "single_latent_seq_len",
            "height",
            "width",
            "inference_steps",
        ]

        results = []

        for _, group in df.groupby(group_cols):
            # Find baseline configuration (ulysses_degree=1, ring_attention_degree=1)
            baseline = group[(group["ulysses_degree"] == 1) & (group["ring_attention_degree"] == 1)]

            if len(baseline) == 0:  # Skip if no baseline configuration
                continue

            baseline_duration = baseline.iloc[0]["Total Duration"]

            # Compare each configuration with the baseline
            for _, row in group.iterrows():
                if row["ulysses_degree"] == 1 and row["ring_attention_degree"] == 1:
                    continue  # Skip baseline configuration

                # Calculate scaling metrics
                degree_ratio = row["ulysses_degree"] * row["ring_attention_degree"]  # Compare to baseline (1*1)
                actual_speedup = baseline_duration / row["Total Duration"]
                scaling_factor = actual_speedup / degree_ratio

                if actual_speedup < 1:
                    logger.info(
                        f"Actual speedup is less than 1 for ulysses={row['ulysses_degree']}, ring={row['ring_attention_degree']} for bs{row['batch_size']} seq_len{row['single_latent_seq_len']} if_torch_compile{row['if_torch_compile']}"
                    )
                    continue

                if scaling_factor > 1:
                    logger.info(
                        f"Scaling factor is greater than 1 for ulysses={row['ulysses_degree']}, ring={row['ring_attention_degree']} for bs{row['batch_size']} seq_len{row['single_latent_seq_len']} if_torch_compile{row['if_torch_compile']}"
                    )
                    continue

                results.append(
                    {
                        "batch_size": row["batch_size"],
                        "height": row["height"],
                        "width": row["width"],
                        "seq_len": row["single_latent_seq_len"],
                        "if_torch_compile": row["if_torch_compile"],
                        "ulysses_degree": row["ulysses_degree"],
                        "ring_degree": row["ring_attention_degree"],
                        "baseline_duration": baseline_duration,
                        "duration_per_step": row["Total Duration"] / row["inference_steps"],
                        "actual_speedup": actual_speedup,
                        "ideal_speedup": degree_ratio,
                        "scaling_factor": scaling_factor,
                    }
                )

        return pd.DataFrame(results)

    def find_best_degrees(
        self, target_batch_size, target_height, target_width, target_num_inference_steps, latency_threshold=None
    ) -> ScalingConfig | None:
        results = self.df
        filtered_df = results[
            (results["batch_size"] == target_batch_size)
            & (
                (results["height"] == target_height) & (results["width"] == target_width)
                | (results["height"] == target_width) & (results["width"] == target_height)
            )
        ]
        if filtered_df.empty:
            # find the closest by seq_len
            target_seq_len = target_height * target_width // 16 // 16
            same_batch = results[results["batch_size"] == target_batch_size]
            closest_idx = (same_batch["seq_len"] - target_seq_len).abs().idxmin()
            filtered_df = same_batch.loc[[closest_idx]]
            logger.info(
                f"Approximating {target_width}x{target_height} to {filtered_df.iloc[0]['width']}x{filtered_df.iloc[0]['height']}"
            )

        if filtered_df.empty:
            raise ValueError("Batch size {} not found in the dataset".format(target_batch_size))

        best_scaling_factor = 0
        best_config = None
        min_gpus_config = None
        min_gpus = float("inf")
        min_duration = float("inf")
        min_duration_config = None

        for _, row in filtered_df.iterrows():
            scaling_factor = row["scaling_factor"]
            actual_speedup = row["actual_speedup"]
            duration_per_step = row["duration_per_step"]
            total_gpus = row["ulysses_degree"] * row["ring_degree"]

            # Track configuration with minimum duration
            if duration_per_step * target_num_inference_steps < min_duration:
                min_duration = duration_per_step * target_num_inference_steps
                min_duration_config = ScalingConfig(
                    ulysses_degree=row["ulysses_degree"],
                    ring_degree=row["ring_degree"],
                    scaling_factor=scaling_factor,
                    actual_speedup=actual_speedup,
                    batch_size=row["batch_size"],
                    seq_len=row["seq_len"],
                    estimated_time=min_duration,
                )

            # Check if this configuration meets performance requirements
            if actual_speedup >= 1 and scaling_factor <= 1:  # Valid scaling case
                if latency_threshold is None:
                    # Without latency constraint, optimize for scaling efficiency
                    if scaling_factor > best_scaling_factor:
                        best_scaling_factor = scaling_factor
                        best_config = ScalingConfig(
                            ulysses_degree=row["ulysses_degree"],
                            ring_degree=row["ring_degree"],
                            scaling_factor=scaling_factor,
                            actual_speedup=actual_speedup,
                            batch_size=row["batch_size"],
                            seq_len=row["seq_len"],
                            estimated_time=duration_per_step * target_num_inference_steps,
                        )
                else:
                    # With latency constraint, find minimum GPUs that meet the threshold
                    if duration_per_step * target_num_inference_steps <= latency_threshold and total_gpus < min_gpus:
                        min_gpus = total_gpus
                        min_gpus_config = ScalingConfig(
                            ulysses_degree=row["ulysses_degree"],
                            ring_degree=row["ring_degree"],
                            scaling_factor=scaling_factor,
                            actual_speedup=actual_speedup,
                            batch_size=row["batch_size"],
                            seq_len=row["seq_len"],
                            estimated_time=duration_per_step * target_num_inference_steps,
                        )

        # if best_config is None:
        #     best_config = ScalingConfig(
        #         ulysses_degree=4,
        #         ring_degree=1,
        #         scaling_factor=1,
        #         actual_speedup=1,
        #         batch_size=target_batch_size,
        #         seq_len=target_height * target_width,
        #         estimated_time=min_duration,
        #     )

        # Return logic:
        # 1. If latency threshold is specified and we found a valid config, return it
        # 2. If latency threshold is specified but no valid config found, return the fastest config
        # 3. If no latency threshold, return the best scaling config
        if latency_threshold is not None:
            return min_gpus_config if min_gpus_config is not None else min_duration_config
        return best_config
