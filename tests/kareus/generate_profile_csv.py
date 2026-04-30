"""Generate profile CSV from Bayesian Optimization JSONL logs and nonpartition prepost profiling.

Unified script supporting both non-CP (cp=1) and CP (cp>1) modes.
- Non-CP: loads fwd_attn, fwd_mlp, bwd_attn, bwd_mlp partitions
- CP: loads 5 fwd + 7 bwd CP sub-partitions (qkv_ar, qkv_ag, ao_ag, ao_ar, mlp, etc.)

Reads BO results from:
  {bayesian_dir}/logs/{model}/cp{cp}-tp{tp}-bs{bs}-seq{seq}/{partition_name}/eval_results.jsonl

Reads nonpartition prepost results from:
  {prepost_dir}/logs/{model}/cp{cp}-tp{tp}-bs{bs}-seq{seq}/nonpartition/{freq}/(preprocess|postprocess|loss)_energy.csv
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from typing import Any, Literal
import os

import sys

import numpy as np
import pandas as pd
import tyro

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bayesian'))
from common.model_config import get_model_config, GPU_CONFIGS, get_p2p_power


@dataclass
class Args:
    # Name of the model (must exist in model_config.MODEL_REGISTRY)
    model_name: str
    # Tensor-parallel size
    tensor_parallel_size: int
    # Context-parallel size. 1 = non-CP mode, >1 = CP mode
    context_parallel_size: int
    # Number of pipeline-parallel stages
    pipeline_parallel_size: int
    # Micro batch size
    batch_size: int
    # Sequence length
    seq_len: int
    # Directory containing BO results
    bayesian_profile_dir: str = "../bayesian"
    # Directory containing nonpartition prepost profiling results
    prepost_profile_dir: str = "../bayesian"
    # Directory to write the output profile CSV into
    output_dir: str = "."
    # GPU type for p2p_power lookup
    gpu_type: Literal["A40", "A100"] = "A100"
    # Generate backward candidates with recompute-forward configs
    use_activation_checkpointing: bool = True
    # Scale sum_time and sum_energy by 1.15
    scale_time_energy: bool = True


def _read_time_energy_csv(path: str) -> tuple[float, float]:
    """Read a CSV with time/energy columns and return first-row (time, per_rank_energy).

    Expects the CSV written by the nonpartition profiler with columns:
      time (s), total_energy (J), per_rank_energy (J)
    Reads 'per_rank_energy (J)' by name (falls back to column index 2).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.shape[0] < 1 or df.shape[1] < 3:
        raise RuntimeError(f"Malformed CSV at {path} (need at least 1 row and 3 columns)")
    row = df.iloc[0]
    if "per_rank_energy (J)" in df.columns:
        return float(row.iloc[0]), float(row["per_rank_energy (J)"])
    return float(row.iloc[0]), float(row.iloc[2])


def read_prepost_profile(
    prepost_profile_dir: str,
    tensor_parallel_size: int,
    context_parallel_size: int,
    batch_size: int,
    seq_len: int,
    freqs: list[int],
    model_name: str,
):
    """Read pre/post profiling results from nonpartition profiler CSVs.

    Expects files in:
      {prepost_profile_dir}/logs/{model}/cp{cp}-tp{tp}-bs{bs}-seq{seq}/nonpartition/{freq}/(preprocess|postprocess|loss)_energy.csv
    Returns dict keyed by frequency with forward-embedding, forward-output, loss-func,
    backward-embedding, backward-output time/energy pairs.
    """
    results: dict[int, dict[str, tuple[float, float]]] = {}
    for frequency in freqs:
        freq_dir = (
            f"{prepost_profile_dir}/logs/{model_name}"
            f"/cp{context_parallel_size}-tp{tensor_parallel_size}-bs{batch_size}-seq{seq_len}"
            f"/nonpartition/{frequency}"
        )
        emb_path = f"{freq_dir}/preprocess_energy.csv"
        out_path = f"{freq_dir}/postprocess_energy.csv"
        loss_path = f"{freq_dir}/loss_energy.csv"
        emb_bwd_path = f"{freq_dir}/preprocess_backward_energy.csv"
        out_bwd_path = f"{freq_dir}/postprocess_backward_energy.csv"

        emb = _read_time_energy_csv(emb_path)
        out = _read_time_energy_csv(out_path)
        loss = _read_time_energy_csv(loss_path)
        emb_bwd = _read_time_energy_csv(emb_bwd_path)
        out_bwd = _read_time_energy_csv(out_bwd_path)
        results[int(frequency)] = {
            "forward-embedding": emb,
            "forward-output": out,
            "loss-func": loss,
            "backward-embedding": emb_bwd,
            "backward-output": out_bwd,
        }
    return results


def _read_bo_jsonl(
    bayesian_profile_dir: str,
    partition_name: str,
    model_name: str,
    context_parallel_size: int,
    tensor_parallel_size: int,
    batch_size: int,
    seq_len: int,
):
    """Read BO results from JSONL and group by frequency.

    Path: {bayesian_dir}/logs/{model}/cp{cp}-tp{tp}-bs{bs}-seq{seq}/{partition_name}/eval_results.jsonl
    Direction is encoded in partition_name (fwd_*, bwd_*). Always reads eval_results.jsonl.

    Returns dict: { frequency: [((overlap_start, overlap_end, sm, block), (time, energy))] }
    """
    results_by_freq: dict[int, list[tuple[tuple[int, int, int, int], tuple[float, float]]]] = {}
    jsonl_path = (
        f"{bayesian_profile_dir}/logs/{model_name}"
        f"/cp{context_parallel_size}-tp{tensor_parallel_size}-bs{batch_size}-seq{seq_len}"
        f"/{partition_name}/eval_results.jsonl"
    )
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(jsonl_path)

    total = 0
    with open(jsonl_path, "r") as f:
        for line in f:
            ln = line.strip()
            if not ln:
                continue
            try:
                row = json.loads(ln)
            except Exception as exc:
                print(f"Skipping malformed JSON line in {jsonl_path}: {exc}")
                continue

            if not all(k in row for k in ("freq", "overlap_start", "overlap_end", "sm", "block", "time_s", "energy_j")):
                print(f"Skipping malformed JSON line in {jsonl_path}: {row}")
                continue

            frequency = int(row["freq"])
            overlap_start = row["overlap_start"]
            overlap_end = row["overlap_end"]
            sm = int(row["sm"])
            block = int(row["block"])
            time_val = float(row["time_s"])
            energy_val = float(row["energy_j"])

            key = (overlap_start, overlap_end, sm, block)
            value = (time_val, energy_val)
            results_by_freq.setdefault(frequency, []).append((key, value))
            total += 1

    print(f"Loaded {total} BO configs from {jsonl_path} across {len(results_by_freq)} frequencies.")
    return results_by_freq


def pareto_optimal(
    config_results: list[tuple[Any, tuple[float, float]]],
    p2p_power: float,
    min_effective_energy_improvement: float = 1e-4,
) -> list[tuple[Any, tuple[float, float]]]:
    """Filter a list of (config, (time, energy)) to Pareto-optimal ones.

    Effective energy = energy - p2p_power * time.
    Keeps points that strictly improve effective energy by at least the tolerance.
    """
    if not config_results:
        return []

    sorted_items = sorted(config_results, key=lambda x: (x[1][0], x[1][1]))

    pareto: list[tuple[Any, tuple[float, float]]] = []
    best_energy = float("inf")
    for cfg, (t, e) in sorted_items:
        ef_e = e - p2p_power * t
        if ef_e + min_effective_energy_improvement < best_energy:
            pareto.append((cfg, (t, e)))
            best_energy = ef_e
    return pareto


def compute_layers_per_stage(
    num_layers: int,
    pipeline_parallel_size: int,
    num_layers_in_first_pipeline_stage: int,
    num_layers_in_last_pipeline_stage: int,
) -> list[int]:
    """Compute number of layers per pipeline stage with uneven split support."""
    if pipeline_parallel_size <= 0:
        raise ValueError("pipeline_parallel_size must be positive")

    if pipeline_parallel_size == 1:
        return [num_layers]

    first = int(max(0, num_layers_in_first_pipeline_stage))
    last = int(max(0, num_layers_in_last_pipeline_stage))

    if pipeline_parallel_size == 2:
        if first + last != num_layers:
            if first > num_layers:
                warnings.warn(
                    f"First stage layers ({first}) exceed total layers ({num_layers}); clamping to total."
                )
                first = num_layers
            last = max(0, num_layers - first)
            if first + last != num_layers:
                warnings.warn(
                    f"Adjusted first/last layers to [{first}, {last}] to match total layers {num_layers}."
                )
        return [first, last]

    remaining = num_layers - first - last
    if remaining < 0:
        warnings.warn(
            f"first({first}) + last({last}) exceed total layers ({num_layers}); reducing last to fit."
        )
        last = max(0, num_layers - first)
        remaining = num_layers - first - last

    middle_stages = pipeline_parallel_size - 2
    base = remaining // middle_stages if middle_stages > 0 else 0
    remainder = remaining % middle_stages if middle_stages > 0 else 0
    middle = [base + (1 if i < remainder else 0) for i in range(middle_stages)]

    layers = [first] + middle + [last]
    delta = num_layers - sum(layers)
    if delta != 0:
        layers[-1] = max(0, layers[-1] + delta)
    return layers


# ---------------------------------------------------------------------------
# Non-CP mode (cp == 1)
# ---------------------------------------------------------------------------

def _main_non_cp(
    bayesian_profile_dir: str,
    prepost_profile_dir: str,
    tensor_parallel_size: int,
    context_parallel_size: int,
    pipeline_parallel_size: int,
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_layers_in_first_pipeline_stage: int,
    num_layers_in_last_pipeline_stage: int,
    p2p_power: float,
    use_activation_checkpointing: bool,
    model_name: str,
    scale_time_energy: bool,
    output_dir: str = ".",
) -> None:
    """Non-CP mode: 2 fwd partitions (attn, mlp) + 2 bwd partitions."""
    attention_fwd_map = _read_bo_jsonl(
        bayesian_profile_dir, "fwd_attn", model_name,
        context_parallel_size, tensor_parallel_size, batch_size, seq_len,
    )
    mlp_fwd_map = _read_bo_jsonl(
        bayesian_profile_dir, "fwd_mlp", model_name,
        context_parallel_size, tensor_parallel_size, batch_size, seq_len,
    )
    attention_bwd_map = _read_bo_jsonl(
        bayesian_profile_dir, "bwd_attn", model_name,
        context_parallel_size, tensor_parallel_size, batch_size, seq_len,
    )
    mlp_bwd_map = _read_bo_jsonl(
        bayesian_profile_dir, "bwd_mlp", model_name,
        context_parallel_size, tensor_parallel_size, batch_size, seq_len,
    )

    freqs_set = (
        set(attention_fwd_map.keys()) & set(mlp_fwd_map.keys())
        & set(attention_bwd_map.keys()) & set(mlp_bwd_map.keys())
    )
    if not freqs_set:
        raise RuntimeError("No overlapping frequencies found across BO results for attention/mlp forward/backward.")
    freqs = sorted(freqs_set)
    print(f"Frequencies from BO results: {freqs}")

    prepost_profiling_results = read_prepost_profile(
        prepost_profile_dir, tensor_parallel_size, context_parallel_size,
        batch_size, seq_len, freqs, model_name,
    )

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(
        output_dir,
        f"profile_{model_name}_cp{context_parallel_size}_tp{tensor_parallel_size}_bs{batch_size}_seq{seq_len}.csv",
    )
    profile_csv = open(csv_path, "w")
    if use_activation_checkpointing:
        profile_csv.write("stage,instruction,frequency,recompute_attention_configs,recompute_mlp_configs,attention_configs,mlp_configs,time,energy\n")
    else:
        profile_csv.write("stage,instruction,frequency,attention_configs,mlp_configs,time,energy\n")

    layers_per_stage = compute_layers_per_stage(
        num_layers, pipeline_parallel_size,
        num_layers_in_first_pipeline_stage, num_layers_in_last_pipeline_stage,
    )
    print(f"Layers per stage: {layers_per_stage}")

    forward_points_by_stage: dict[int, list] = {s: [] for s in range(pipeline_parallel_size)}
    backward_points_by_stage: dict[int, list] = {s: [] for s in range(pipeline_parallel_size)}

    # Forward candidates
    for frequency in freqs:
        attention_fwd_result = attention_fwd_map.get(frequency, [])
        mlp_fwd_result = mlp_fwd_map.get(frequency, [])

        for attn_config, attn_result in attention_fwd_result:
            for mlp_config, mlp_result in mlp_fwd_result:
                sum_time = attn_result[0] + mlp_result[0]
                sum_energy = attn_result[1] + mlp_result[1]
                if scale_time_energy:
                    sum_time *= 1.15
                    sum_energy *= 1.15

                for stage in range(pipeline_parallel_size):
                    fwd_time = sum_time * layers_per_stage[stage] * 2
                    fwd_energy = sum_energy * layers_per_stage[stage] * 2
                    if stage == 0:
                        fwd_time += prepost_profiling_results[frequency]["forward-embedding"][0]
                        fwd_energy += prepost_profiling_results[frequency]["forward-embedding"][1]
                    elif stage == pipeline_parallel_size - 1:
                        fwd_time += prepost_profiling_results[frequency]["forward-output"][0]
                        fwd_time += prepost_profiling_results[frequency]["loss-func"][0]
                        fwd_energy += prepost_profiling_results[frequency]["forward-output"][1]
                        fwd_energy += prepost_profiling_results[frequency]["loss-func"][1]
                    forward_points_by_stage[stage].append(
                        ((frequency, attn_config, mlp_config), (fwd_time, fwd_energy))
                    )

    if use_activation_checkpointing:
        # Backward candidates with recompute-forward configs
        for frequency in freqs:
            attention_fwd_result = attention_fwd_map.get(frequency, [])
            mlp_fwd_result = mlp_fwd_map.get(frequency, [])
            attention_bwd_result = attention_bwd_map.get(frequency, [])
            mlp_bwd_result = mlp_bwd_map.get(frequency, [])

            for rec_attn_cfg, rec_attn_res in attention_fwd_result:
                for rec_mlp_cfg, rec_mlp_res in mlp_fwd_result:
                    for bwd_attn_cfg, bwd_attn_res in attention_bwd_result:
                        for bwd_mlp_cfg, bwd_mlp_res in mlp_bwd_result:
                            sum_time = rec_attn_res[0] + rec_mlp_res[0] + bwd_attn_res[0] + bwd_mlp_res[0]
                            sum_energy = rec_attn_res[1] + rec_mlp_res[1] + bwd_attn_res[1] + bwd_mlp_res[1]
                            if scale_time_energy:
                                sum_time *= 1.15
                                sum_energy *= 1.15

                            for stage in range(pipeline_parallel_size):
                                bwd_time = sum_time * layers_per_stage[stage] * 2
                                bwd_energy = sum_energy * layers_per_stage[stage] * 2
                                if stage == 0:
                                    bwd_time += prepost_profiling_results[frequency]["backward-embedding"][0] * 2
                                    bwd_energy += prepost_profiling_results[frequency]["backward-embedding"][1] * 2
                                elif stage == pipeline_parallel_size - 1:
                                    bwd_time += prepost_profiling_results[frequency]["backward-output"][0]
                                    bwd_energy += prepost_profiling_results[frequency]["backward-output"][1]
                                backward_points_by_stage[stage].append(
                                    ((frequency, rec_attn_cfg, rec_mlp_cfg, bwd_attn_cfg, bwd_mlp_cfg), (bwd_time, bwd_energy))
                                )
    else:
        # Backward candidates without AC
        for frequency in freqs:
            attention_bwd_result = attention_bwd_map.get(frequency, [])
            mlp_bwd_result = mlp_bwd_map.get(frequency, [])

            for attn_config, attn_result in attention_bwd_result:
                for mlp_config, mlp_result in mlp_bwd_result:
                    sum_time = attn_result[0] + mlp_result[0]
                    sum_energy = attn_result[1] + mlp_result[1]
                    if scale_time_energy:
                        sum_time *= 1.15
                        sum_energy *= 1.15

                    for stage in range(pipeline_parallel_size):
                        bwd_time = sum_time * layers_per_stage[stage] * 2
                        bwd_energy = sum_energy * layers_per_stage[stage] * 2
                        if stage == 0:
                            bwd_time += prepost_profiling_results[frequency]["backward-embedding"][0] * 2
                            bwd_energy += prepost_profiling_results[frequency]["backward-embedding"][1] * 2
                        elif stage == pipeline_parallel_size - 1:
                            bwd_time += prepost_profiling_results[frequency]["backward-output"][0]
                            bwd_energy += prepost_profiling_results[frequency]["backward-output"][1]
                        backward_points_by_stage[stage].append(
                            ((frequency, attn_config, mlp_config), (bwd_time, bwd_energy))
                        )

    # Write Pareto-optimal points
    for stage in range(pipeline_parallel_size):
        fwd_pareto = pareto_optimal(forward_points_by_stage[stage], p2p_power)
        print(f"generated {len(fwd_pareto)}/{len(forward_points_by_stage[stage])} Pareto-optimal forward candidates for stage {stage}")
        if use_activation_checkpointing:
            for (frequency, attn_config, mlp_config), (fwd_time, fwd_energy) in fwd_pareto:
                profile_csv.write(
                    f"{stage},forward,{frequency},,,{'-'.join(map(str, attn_config))},{'-'.join(map(str, mlp_config))},{fwd_time},{fwd_energy}\n"
                )
        else:
            for (frequency, attn_config, mlp_config), (fwd_time, fwd_energy) in fwd_pareto:
                profile_csv.write(
                    f"{stage},forward,{frequency},{'-'.join(map(str, attn_config))},{'-'.join(map(str, mlp_config))},{fwd_time},{fwd_energy}\n"
                )

        bwd_pareto = pareto_optimal(backward_points_by_stage[stage], p2p_power)
        print(f"generated {len(bwd_pareto)}/{len(backward_points_by_stage[stage])} Pareto-optimal backward candidates for stage {stage}")
        if use_activation_checkpointing:
            for (frequency, rec_attn_cfg, rec_mlp_cfg, bwd_attn_cfg, bwd_mlp_cfg), (bwd_time, bwd_energy) in bwd_pareto:
                profile_csv.write(
                    f"{stage},backward,{frequency},"
                    f"{'-'.join(map(str, rec_attn_cfg))},{'-'.join(map(str, rec_mlp_cfg))},"
                    f"{'-'.join(map(str, bwd_attn_cfg))},{'-'.join(map(str, bwd_mlp_cfg))},"
                    f"{bwd_time},{bwd_energy}\n"
                )
        else:
            for (frequency, attn_config, mlp_config), (bwd_time, bwd_energy) in bwd_pareto:
                profile_csv.write(
                    f"{stage},backward,{frequency},{'-'.join(map(str, attn_config))},{'-'.join(map(str, mlp_config))},{bwd_time},{bwd_energy}\n"
                )

    profile_csv.close()
    print(f"Profile CSV saved: {csv_path}")


# ---------------------------------------------------------------------------
# CP mode (cp > 1)
# ---------------------------------------------------------------------------

def _main_cp(
    bayesian_profile_dir: str,
    prepost_profile_dir: str,
    tensor_parallel_size: int,
    context_parallel_size: int,
    pipeline_parallel_size: int,
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_layers_in_first_pipeline_stage: int,
    num_layers_in_last_pipeline_stage: int,
    p2p_power: float,
    use_activation_checkpointing: bool,
    model_name: str,
    scale_time_energy: bool,
    output_dir: str = ".",
) -> None:
    """CP mode: 5 fwd + 7 bwd CP sub-partitions."""
    common = dict(
        bayesian_profile_dir=bayesian_profile_dir,
        model_name=model_name,
        context_parallel_size=context_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    fwd_qkv_ar_map = _read_bo_jsonl(partition_name="fwd_qkv_ar", **common)
    fwd_qkv_ag_map = _read_bo_jsonl(partition_name="fwd_qkv_ag", **common)
    fwd_ao_ag_map = _read_bo_jsonl(partition_name="fwd_ao_ag", **common)
    fwd_ao_ar_map = _read_bo_jsonl(partition_name="fwd_ao_ar", **common)
    fwd_mlp_map = _read_bo_jsonl(partition_name="fwd_mlp", **common)
    bwd_qkv_ar_map = _read_bo_jsonl(partition_name="bwd_qkv_ar", **common)
    bwd_qkv_rs_map = _read_bo_jsonl(partition_name="bwd_qkv_rs", **common)
    bwd_a_rs_map = _read_bo_jsonl(partition_name="bwd_a_rs", **common)
    bwd_a_ag_map = _read_bo_jsonl(partition_name="bwd_a_ag", **common)
    bwd_o_ag_map = _read_bo_jsonl(partition_name="bwd_o_ag", **common)
    bwd_o_ar_map = _read_bo_jsonl(partition_name="bwd_o_ar", **common)
    bwd_mlp_map = _read_bo_jsonl(partition_name="bwd_mlp", **common)

    freqs_set = (
        set(fwd_qkv_ar_map.keys()) & set(fwd_qkv_ag_map.keys())
        & set(fwd_ao_ag_map.keys()) & set(fwd_ao_ar_map.keys())
        & set(fwd_mlp_map.keys()) & set(bwd_qkv_ar_map.keys())
        & set(bwd_qkv_rs_map.keys()) & set(bwd_a_rs_map.keys())
        & set(bwd_a_ag_map.keys()) & set(bwd_o_ag_map.keys())
        & set(bwd_o_ar_map.keys()) & set(bwd_mlp_map.keys())
    )
    if not freqs_set:
        raise RuntimeError("No overlapping frequencies found across all CP partitions.")
    freqs = sorted(freqs_set)
    print(f"Frequencies from BO results: {freqs}")

    prepost_profiling_results = read_prepost_profile(
        prepost_profile_dir, tensor_parallel_size, context_parallel_size,
        batch_size, seq_len, freqs, model_name,
    )

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(
        output_dir,
        f"profile_{model_name}_cp{context_parallel_size}_tp{tensor_parallel_size}_bs{batch_size}_seq{seq_len}.csv",
    )
    profile_csv = open(csv_path, "w")
    profile_csv.write(
        "stage,instruction,frequency,"
        "fwd_qkv_ar,fwd_qkv_ag,fwd_ao_ag,fwd_ao_ar,fwd_mlp,"
        "bwd_qkv_ar,bwd_qkv_rs,bwd_a_rs,bwd_a_ag,bwd_o_ag,bwd_o_ar,bwd_mlp,"
        "time,energy\n"
    )

    layers_per_stage = compute_layers_per_stage(
        num_layers, pipeline_parallel_size,
        num_layers_in_first_pipeline_stage, num_layers_in_last_pipeline_stage,
    )
    print(f"Layers per stage: {layers_per_stage}")

    # Pareto-filter each partition per frequency before combination
    def _filter_map(m):
        return {f: pareto_optimal(items, p2p_power, 1e-4) for f, items in m.items()}

    fwd_qkv_ar_map_f = _filter_map(fwd_qkv_ar_map)
    fwd_qkv_ag_map_f = _filter_map(fwd_qkv_ag_map)
    fwd_ao_ag_map_f = _filter_map(fwd_ao_ag_map)
    fwd_ao_ar_map_f = _filter_map(fwd_ao_ar_map)
    fwd_mlp_map_f = _filter_map(fwd_mlp_map)
    bwd_qkv_ar_map_f = _filter_map(bwd_qkv_ar_map)
    bwd_qkv_rs_map_f = _filter_map(bwd_qkv_rs_map)
    bwd_a_rs_map_f = _filter_map(bwd_a_rs_map)
    bwd_a_ag_map_f = _filter_map(bwd_a_ag_map)
    bwd_o_ag_map_f = _filter_map(bwd_o_ag_map)
    bwd_o_ar_map_f = _filter_map(bwd_o_ar_map)
    bwd_mlp_map_f = _filter_map(bwd_mlp_map)

    forward_points_by_stage: dict[int, list] = {s: [] for s in range(pipeline_parallel_size)}
    backward_points_by_stage: dict[int, list] = {s: [] for s in range(pipeline_parallel_size)}

    # Forward candidates (5 partitions, MLP doubled)
    for frequency in freqs:
        qkv_ar_list = fwd_qkv_ar_map_f.get(frequency, [])
        qkv_ag_list = fwd_qkv_ag_map_f.get(frequency, [])
        ao_ag_list = fwd_ao_ag_map_f.get(frequency, [])
        ao_ar_list = fwd_ao_ar_map_f.get(frequency, [])
        mlp_fwd_list = fwd_mlp_map_f.get(frequency, [])

        for (cfg_qkv_ar, (t1, e1)) in qkv_ar_list:
            for (cfg_qkv_ag, (t2, e2)) in qkv_ag_list:
                for (cfg_ao_ag, (t3, e3)) in ao_ag_list:
                    for (cfg_ao_ar, (t4, e4)) in ao_ar_list:
                        for (cfg_mlp_fwd, (t5, e5)) in mlp_fwd_list:
                            sum_time = t1 + t2 + t3 + t4 + t5 * 2
                            sum_energy = e1 + e2 + e3 + e4 + e5 * 2
                            if scale_time_energy:
                                sum_time *= 1.15
                                sum_energy *= 1.15

                            for stage in range(pipeline_parallel_size):
                                fwd_time = sum_time * layers_per_stage[stage]
                                fwd_energy = sum_energy * layers_per_stage[stage]
                                if stage == 0:
                                    fwd_time += prepost_profiling_results[frequency]["forward-embedding"][0]
                                    fwd_energy += prepost_profiling_results[frequency]["forward-embedding"][1]
                                elif stage == pipeline_parallel_size - 1:
                                    fwd_time += prepost_profiling_results[frequency]["forward-output"][0]
                                    fwd_time += prepost_profiling_results[frequency]["loss-func"][0]
                                    fwd_energy += prepost_profiling_results[frequency]["forward-output"][1]
                                    fwd_energy += prepost_profiling_results[frequency]["loss-func"][1]
                                forward_points_by_stage[stage].append((
                                    (frequency, cfg_qkv_ar, cfg_qkv_ag, cfg_ao_ag, cfg_ao_ar, cfg_mlp_fwd),
                                    (fwd_time, fwd_energy),
                                ))

    if use_activation_checkpointing:
        # Backward with AC: 5 recompute-fwd + 7 bwd partitions
        for frequency in freqs:
            qkv_ar_list_f = fwd_qkv_ar_map_f.get(frequency, [])
            qkv_ag_list_f = fwd_qkv_ag_map_f.get(frequency, [])
            ao_ag_list_f = fwd_ao_ag_map_f.get(frequency, [])
            ao_ar_list_f = fwd_ao_ar_map_f.get(frequency, [])
            mlp_fwd_list = fwd_mlp_map_f.get(frequency, [])

            qkv_ar_list_b = bwd_qkv_ar_map_f.get(frequency, [])
            qkv_rs_list_b = bwd_qkv_rs_map_f.get(frequency, [])
            a_rs_list_b = bwd_a_rs_map_f.get(frequency, [])
            a_ag_list_b = bwd_a_ag_map_f.get(frequency, [])
            o_ag_list_b = bwd_o_ag_map_f.get(frequency, [])
            o_ar_list_b = bwd_o_ar_map_f.get(frequency, [])
            mlp_bwd_list = bwd_mlp_map_f.get(frequency, [])

            for (cfg_f_qkv_ar, (t1, e1)) in qkv_ar_list_f:
                for (cfg_f_qkv_ag, (t2, e2)) in qkv_ag_list_f:
                    for (cfg_f_ao_ag, (t3, e3)) in ao_ag_list_f:
                        for (cfg_f_ao_ar, (t4, e4)) in ao_ar_list_f:
                            for (cfg_f_mlp, (t5, e5)) in mlp_fwd_list:
                                for (cfg_b_qkv_ar, (tb1, eb1)) in qkv_ar_list_b:
                                    for (cfg_b_qkv_rs, (tb2, eb2)) in qkv_rs_list_b:
                                        for (cfg_b_a_rs, (tb3, eb3)) in a_rs_list_b:
                                            for (cfg_b_a_ag, (tb4, eb4)) in a_ag_list_b:
                                                for (cfg_b_o_ag, (tb5, eb5)) in o_ag_list_b:
                                                    for (cfg_b_o_ar, (tb6, eb6)) in o_ar_list_b:
                                                        for (cfg_b_mlp, (tb7, eb7)) in mlp_bwd_list:
                                                            sum_time = (t1 + t2 + t3 + t4 + t5 * 2 + tb1 + tb2 + tb3 + tb4 + tb5 + tb6 + tb7 * 2)
                                                            sum_energy = (e1 + e2 + e3 + e4 + e5 * 2 + eb1 + eb2 + eb3 + eb4 + eb5 + eb6 + eb7 * 2)
                                                            if scale_time_energy:
                                                                sum_time *= 1.15
                                                                sum_energy *= 1.15

                                                            for stage in range(pipeline_parallel_size):
                                                                bwd_time = sum_time * layers_per_stage[stage]
                                                                bwd_energy = sum_energy * layers_per_stage[stage]
                                                                if stage == 0:
                                                                    bwd_time += prepost_profiling_results[frequency]["backward-embedding"][0] * 2
                                                                    bwd_energy += prepost_profiling_results[frequency]["backward-embedding"][1] * 2
                                                                elif stage == pipeline_parallel_size - 1:
                                                                    bwd_time += prepost_profiling_results[frequency]["backward-output"][0]
                                                                    bwd_energy += prepost_profiling_results[frequency]["backward-output"][1]
                                                                backward_points_by_stage[stage].append((
                                                                    (frequency, cfg_f_qkv_ar, cfg_f_qkv_ag, cfg_f_ao_ag, cfg_f_ao_ar, cfg_f_mlp,
                                                                     cfg_b_qkv_ar, cfg_b_qkv_rs, cfg_b_a_rs, cfg_b_a_ag, cfg_b_o_ag, cfg_b_o_ar, cfg_b_mlp),
                                                                    (bwd_time, bwd_energy),
                                                                ))
    else:
        # Backward without AC: 7 bwd partitions only
        for frequency in freqs:
            qkv_ar_list_b = bwd_qkv_ar_map_f.get(frequency, [])
            qkv_rs_list_b = bwd_qkv_rs_map_f.get(frequency, [])
            a_rs_list_b = bwd_a_rs_map_f.get(frequency, [])
            a_ag_list_b = bwd_a_ag_map_f.get(frequency, [])
            o_ag_list_b = bwd_o_ag_map_f.get(frequency, [])
            o_ar_list_b = bwd_o_ar_map_f.get(frequency, [])
            mlp_bwd_list = bwd_mlp_map_f.get(frequency, [])

            for (cfg_b_qkv_ar, (t1, e1)) in qkv_ar_list_b:
                for (cfg_b_qkv_rs, (t2, e2)) in qkv_rs_list_b:
                    for (cfg_b_a_rs, (t3, e3)) in a_rs_list_b:
                        for (cfg_b_a_ag, (t4, e4)) in a_ag_list_b:
                            for (cfg_b_o_ag, (t5, e5)) in o_ag_list_b:
                                for (cfg_b_o_ar, (t6, e6)) in o_ar_list_b:
                                    for (cfg_b_mlp, (t7, e7)) in mlp_bwd_list:
                                        sum_time = t1 + t2 + t3 + t4 + t5 + t6 + t7 * 2
                                        sum_energy = e1 + e2 + e3 + e4 + e5 + e6 + e7 * 2
                                        if scale_time_energy:
                                            sum_time *= 1.15
                                            sum_energy *= 1.15

                                        for stage in range(pipeline_parallel_size):
                                            bwd_time = sum_time * layers_per_stage[stage]
                                            bwd_energy = sum_energy * layers_per_stage[stage]
                                            if stage == 0:
                                                bwd_time += prepost_profiling_results[frequency]["backward-embedding"][0] * 2
                                                bwd_energy += prepost_profiling_results[frequency]["backward-embedding"][1] * 2
                                            elif stage == pipeline_parallel_size - 1:
                                                bwd_time += prepost_profiling_results[frequency]["backward-output"][0]
                                                bwd_energy += prepost_profiling_results[frequency]["backward-output"][1]
                                            backward_points_by_stage[stage].append((
                                                (frequency, cfg_b_qkv_ar, cfg_b_qkv_rs, cfg_b_a_rs, cfg_b_a_ag, cfg_b_o_ag, cfg_b_o_ar, cfg_b_mlp),
                                                (bwd_time, bwd_energy),
                                            ))

    # Write Pareto-optimal points
    for stage in range(pipeline_parallel_size):
        fwd_pareto = pareto_optimal(forward_points_by_stage[stage], p2p_power)
        print(f"generated {len(fwd_pareto)}/{len(forward_points_by_stage[stage])} Pareto-optimal forward candidates for stage {stage}")
        for cfg_payload, (fwd_time, fwd_energy) in fwd_pareto:
            frequency, f_qkv_ar, f_qkv_ag, f_ao_ag, f_ao_ar, f_mlp = cfg_payload
            profile_csv.write(
                f"{stage},forward,{frequency},"
                f"{'-'.join(map(str, f_qkv_ar))},{'-'.join(map(str, f_qkv_ag))},{'-'.join(map(str, f_ao_ag))},{'-'.join(map(str, f_ao_ar))},{'-'.join(map(str, f_mlp))},"
                f",,,,,,,{fwd_time},{fwd_energy}\n"
            )

        bwd_pareto = pareto_optimal(backward_points_by_stage[stage], p2p_power, 0.01)
        print(f"generated {len(bwd_pareto)}/{len(backward_points_by_stage[stage])} Pareto-optimal backward candidates for stage {stage}")
        if use_activation_checkpointing:
            for cfg_payload, (bwd_time, bwd_energy) in bwd_pareto:
                (frequency, f_qkv_ar, f_qkv_ag, f_ao_ag, f_ao_ar, f_mlp,
                 b_qkv_ar, b_qkv_rs, b_a_rs, b_a_ag, b_o_ag, b_o_ar, b_mlp) = cfg_payload
                profile_csv.write(
                    f"{stage},backward,{frequency},"
                    f"{'-'.join(map(str, f_qkv_ar))},{'-'.join(map(str, f_qkv_ag))},{'-'.join(map(str, f_ao_ag))},{'-'.join(map(str, f_ao_ar))},{'-'.join(map(str, f_mlp))},"
                    f"{'-'.join(map(str, b_qkv_ar))},{'-'.join(map(str, b_qkv_rs))},{'-'.join(map(str, b_a_rs))},{'-'.join(map(str, b_a_ag))},{'-'.join(map(str, b_o_ag))},{'-'.join(map(str, b_o_ar))},{'-'.join(map(str, b_mlp))},"
                    f"{bwd_time},{bwd_energy}\n"
                )
        else:
            for cfg_payload, (bwd_time, bwd_energy) in bwd_pareto:
                (frequency, b_qkv_ar, b_qkv_rs, b_a_rs, b_a_ag, b_o_ag, b_o_ar, b_mlp) = cfg_payload
                profile_csv.write(
                    f"{stage},backward,{frequency},,,,,"
                    f"{'-'.join(map(str, b_qkv_ar))},{'-'.join(map(str, b_qkv_rs))},{'-'.join(map(str, b_a_rs))},{'-'.join(map(str, b_a_ag))},{'-'.join(map(str, b_o_ag))},{'-'.join(map(str, b_o_ar))},{'-'.join(map(str, b_mlp))},"
                    f"{bwd_time},{bwd_energy}\n"
                )

    profile_csv.close()
    print(f"Profile CSV saved: {csv_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = tyro.cli(Args)

    model_cfg = get_model_config(args.model_name)
    num_layers = model_cfg.num_layers
    p2p_power = get_p2p_power(args.gpu_type)

    pp = args.pipeline_parallel_size
    first_stage_layers = model_cfg.num_layers_in_first_pipeline_stage
    last_stage_layers = model_cfg.num_layers_in_last_pipeline_stage

    common_kwargs = dict(
        bayesian_profile_dir=args.bayesian_profile_dir,
        prepost_profile_dir=args.prepost_profile_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        context_parallel_size=args.context_parallel_size,
        pipeline_parallel_size=pp,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_layers=num_layers,
        num_layers_in_first_pipeline_stage=first_stage_layers,
        num_layers_in_last_pipeline_stage=last_stage_layers,
        p2p_power=p2p_power,
        use_activation_checkpointing=args.use_activation_checkpointing,
        model_name=args.model_name,
        scale_time_energy=args.scale_time_energy,
        output_dir=args.output_dir,
    )

    if args.context_parallel_size == 1:
        print("Running in non-CP mode (context_parallel_size=1)")
        _main_non_cp(**common_kwargs)
    else:
        print(f"Running in CP mode (context_parallel_size={args.context_parallel_size})")
        _main_cp(**common_kwargs)
