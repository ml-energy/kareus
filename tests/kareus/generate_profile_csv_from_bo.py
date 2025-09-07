"""Post-process time and energy profiling results from decoupled profiling mode."""

from __future__ import annotations

import argparse
import warnings
from glob import glob
from typing import Literal, Any
import os

import numpy as np
import pandas as pd


def _read_time_energy_csv(path: str, tensor_parallel_size: int) -> tuple[float, float]:
    """Read a simple CSV with time/energy columns and return first-row (time, energy).

    Accepts varying header names; reads by column position (0: time, 1: energy).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.shape[0] < 1 or df.shape[1] < 2:
        raise RuntimeError(f"Malformed CSV at {path} (need at least 1 row and 2 columns)")
    row = df.iloc[0]
    return float(row.iloc[0]), float(row.iloc[1]) / tensor_parallel_size


def read_prepost_profile(
    prepost_profile_dir: str,
    tensor_parallel_size: int,
    batch_size: int,
    seq_len: int,
    freqs: list[int],
):
    """Read pre/post profiling results produced by profile_preprocess/postprocess/loss scripts.

    Expects files in:
      {prepost_profile_dir}/logs/tp{tp}-bs{bs}-seq{seq}/{freq}/(preprocess|postprocess|loss)_energy.csv
      {prepost_profile_dir}/logs/tp{tp}-bs{bs}-seq{seq}/{freq}/(preprocess|postprocess)_backward_energy.csv
    Returns a dict keyed by frequency: {
      freq: {
        "forward-embedding": (time, energy),
        "forward-output": (time, energy),
        "loss-func": (time, energy),
        "backward-embedding": (time, energy),
        "backward-output": (time, energy),
      }
    }
    """
    results: dict[int, dict[str, tuple[float, float]]] = {}
    for frequency in freqs:
        freq_dir = f"{prepost_profile_dir}/logs/tp{tensor_parallel_size}-bs{batch_size}-seq{seq_len}/{frequency}"
        emb_path = f"{freq_dir}/preprocess_energy.csv"
        out_path = f"{freq_dir}/postprocess_energy.csv"
        loss_path = f"{freq_dir}/loss_energy.csv"
        emb_bwd_path = f"{freq_dir}/preprocess_backward_energy.csv"
        out_bwd_path = f"{freq_dir}/postprocess_backward_energy.csv"

        emb = _read_time_energy_csv(emb_path, tensor_parallel_size)
        out = _read_time_energy_csv(out_path, tensor_parallel_size)
        loss = _read_time_energy_csv(loss_path, tensor_parallel_size)
        emb_bwd = _read_time_energy_csv(emb_bwd_path, tensor_parallel_size)
        out_bwd = _read_time_energy_csv(out_bwd_path, tensor_parallel_size)
        results[int(frequency)] = {
            "forward-embedding": emb,
            "forward-output": out,
            "loss-func": loss,
            "backward-embedding": emb_bwd,
            "backward-output": out_bwd,
        }
    return results


def read_bo_partition_results(
    bayesian_profile_dir: str,
    partition: str,
    direction: str,
    tensor_parallel_size: int,
    batch_size: int,
    seq_len: int,
):
    """Read BO results from results_all.csv and group by frequency.

    Path example:
      {bayesian_profile_dir}/{partition}/logs/tp{tp}-bs{bs}-seq{seq}/{direction}/results_all.csv
    CSV columns expected: frequency,overlap_start,overlap_end,comm_sm_number,comm_block_size,time_s,avg_energy_J,(...)

    Returns dict: { frequency: [((overlap_start, overlap_end, comm_sm_number, comm_block_size), (time, energy))] }
    """
    results_by_freq: dict[int, list[tuple[tuple[int, int, int, int], tuple[float, float]]]] = {}
    base_dir = f"{bayesian_profile_dir}/{partition}/logs/tp{tensor_parallel_size}-bs{batch_size}-seq{seq_len}/{direction}"
    csv_path = f"{base_dir}/results_all.csv"
    if not os.path.exists(csv_path):
        print(f"Warning: BO results file not found: {csv_path}")
        return results_by_freq

    df = pd.read_csv(csv_path)
    if 'frequency' not in df.columns or 'time_s' not in df.columns or 'avg_energy_J' not in df.columns:
        raise RuntimeError(f"Malformed BO CSV: missing required columns in {csv_path}")

    for _, row in df.iterrows():
        frequency = int(row['frequency'])
        overlap_start = int(row['overlap_start'])
        overlap_end = int(row['overlap_end'])
        comm_sm_number = int(row['comm_sm_number'])
        comm_block_size = int(row['comm_block_size'])
        time_val = float(row['time_s'])
        energy_val = float(row['avg_energy_J'])

        key = (overlap_start, overlap_end, comm_sm_number, comm_block_size)
        value = (time_val, energy_val)
        results_by_freq.setdefault(frequency, []).append((key, value))

    print(f"Loaded {sum(len(v) for v in results_by_freq.values())} BO configs from {csv_path} across {len(results_by_freq)} frequencies.")
    return results_by_freq


def pareto_optimal(config_results: list[tuple[Any, tuple[float, float]]], p2p_power: float) -> list[tuple[Any, tuple[float, float]]]:
    """Filter a list of (config, (time, energy)) to Pareto-optimal ones.

    A point a dominates b if a.time <= b.time and a.energy <= b.energy and at least one strict.
    Works for any hashable config payload (frequency, overlaps, etc.).
    """
    if not config_results:
        return []

    # Sort by time asc, then energy asc for efficient sweep
    sorted_items = sorted(config_results, key=lambda x: (x[1][0], x[1][1]))

    pareto: list[tuple[tuple[int, int, int, int], tuple[float, float]]] = []
    best_energy = float("inf")
    for cfg, (t, e) in sorted_items:
        # Map the cost to be effective computation energy.
        ef_e = e - p2p_power * t
        if ef_e < best_energy:  # strictly better in energy at this time
            pareto.append((cfg, (t, e)))
            best_energy = ef_e
    return pareto


def main(
    bayesian_profile_dir: str,
    prepost_profile_dir: str,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    batch_size: int,
    seq_len: int,
    num_layers: int,
    p2p_power: float,
) -> None:
    """Run the main routine."""
    print(f"Processing BO partition results in {bayesian_profile_dir} and pre/post results in {prepost_profile_dir}.")

    # Load BO partitioning results grouped by frequency
    attention_fwd_map = read_bo_partition_results(
        bayesian_profile_dir, "attention", "forward",
        tensor_parallel_size, batch_size, seq_len,
    )
    mlp_fwd_map = read_bo_partition_results(
        bayesian_profile_dir, "mlp", "forward",
        tensor_parallel_size, batch_size, seq_len,
    )
    attention_bwd_map = read_bo_partition_results(
        bayesian_profile_dir, "attention", "backward",
        tensor_parallel_size, batch_size, seq_len,
    )
    mlp_bwd_map = read_bo_partition_results(
        bayesian_profile_dir, "mlp", "backward",
        tensor_parallel_size, batch_size, seq_len,
    )

    # Determine usable frequencies as intersection across all partitions/directions
    freqs_set = set(attention_fwd_map.keys()) & set(mlp_fwd_map.keys()) & set(attention_bwd_map.keys()) & set(mlp_bwd_map.keys())
    if not freqs_set:
        raise RuntimeError("No overlapping frequencies found across BO results for attention/mlp forward/backward.")
    freqs = sorted(freqs_set)
    print(f"Frequencies from BO results: {freqs}")

    # Read pre/post process results directly from profiler CSVs for these frequencies
    prepost_profiling_results = read_prepost_profile(
        prepost_profile_dir,
        tensor_parallel_size,
        batch_size,
        seq_len,
        freqs,
    )

    profile_csv = open("profile.csv", "w")
    profile_csv.write("stage,instruction,frequency,attention_configs,mlp_configs,time,energy\n")
    num_layer_per_stage = num_layers // pipeline_parallel_size

    # Accumulate all candidate points across frequencies to filter globally per stage.
    forward_points_by_stage: dict[int, list[tuple[tuple[int, tuple[int, int, int, int], tuple[int, int, int, int]], tuple[float, float]]]] = {s: [] for s in range(pipeline_parallel_size)}
    backward_points_by_stage: dict[int, list[tuple[tuple[int, tuple[int, int, int, int], tuple[int, int, int, int]], tuple[float, float]]]] = {s: [] for s in range(pipeline_parallel_size)}

    # forward candidates
    for frequency in freqs:
        attention_fwd_result = pareto_optimal(attention_fwd_map.get(frequency, []), p2p_power)
        mlp_fwd_result = pareto_optimal(mlp_fwd_map.get(frequency, []), p2p_power)

        for attn_config, attn_result in attention_fwd_result:
            for mlp_config, mlp_result in mlp_fwd_result:
                base_time = (attn_result[0] + mlp_result[0]) * num_layer_per_stage * 2  # 2 nanobatches
                base_energy = (attn_result[1] + mlp_result[1]) * num_layer_per_stage * 2

                for stage in range(pipeline_parallel_size):
                    fwd_time = base_time
                    fwd_energy = base_energy
                    if stage == 0:
                        fwd_time += prepost_profiling_results[frequency]["forward-embedding"][0]
                        fwd_energy += prepost_profiling_results[frequency]["forward-embedding"][1]
                    elif stage == pipeline_parallel_size - 1:
                        fwd_time += prepost_profiling_results[frequency]["forward-output"][0]
                        fwd_time += prepost_profiling_results[frequency]["loss-func"][0]
                        fwd_energy += prepost_profiling_results[frequency]["forward-output"][1]
                        fwd_energy += prepost_profiling_results[frequency]["loss-func"][1]
                    forward_points_by_stage[stage].append(((frequency, attn_config, mlp_config), (fwd_time, fwd_energy)))

    # backward candidates
    for frequency in freqs:
        attention_bwd_result = pareto_optimal(attention_bwd_map.get(frequency, []), p2p_power)
        mlp_bwd_result = pareto_optimal(mlp_bwd_map.get(frequency, []), p2p_power)

        for attn_config, attn_result in attention_bwd_result:
            for mlp_config, mlp_result in mlp_bwd_result:
                base_time = (attn_result[0] + mlp_result[0]) * num_layer_per_stage * 2  # 2 nanobatches
                base_energy = (attn_result[1] + mlp_result[1]) * num_layer_per_stage * 2

                for stage in range(pipeline_parallel_size):
                    bwd_time = base_time
                    bwd_energy = base_energy
                    if stage == 0:
                        bwd_time += prepost_profiling_results[frequency]["backward-embedding"][0]
                        bwd_energy += prepost_profiling_results[frequency]["backward-embedding"][1]
                    elif stage == pipeline_parallel_size - 1:
                        bwd_time += prepost_profiling_results[frequency]["backward-output"][0]
                        bwd_energy += prepost_profiling_results[frequency]["backward-output"][1]
                    backward_points_by_stage[stage].append(((frequency, attn_config, mlp_config), (bwd_time, bwd_energy)))

    # Write only globally Pareto-optimal points across frequency+configs per stage/instruction.
    for stage in range(pipeline_parallel_size):
        # Forward
        fwd_pareto = pareto_optimal(forward_points_by_stage[stage], p2p_power)
        print(f"generated {len(fwd_pareto)}/{len(forward_points_by_stage[stage])} Pareto-optimal forward candidates for stage {stage}")
        for (frequency, attn_config, mlp_config), (fwd_time, fwd_energy) in fwd_pareto:
            profile_csv.write(f"{stage},forward,{frequency},{'-'.join(map(str, attn_config))},{'-'.join(map(str, mlp_config))},{fwd_time},{fwd_energy}\n")

        # Backward
        bwd_pareto = pareto_optimal(backward_points_by_stage[stage], p2p_power)
        print(f"generated {len(bwd_pareto)}/{len(backward_points_by_stage[stage])} Pareto-optimal backward candidates for stage {stage}")
        for (frequency, attn_config, mlp_config), (bwd_time, bwd_energy) in bwd_pareto:
            profile_csv.write(f"{stage},backward,{frequency},{'-'.join(map(str, attn_config))},{'-'.join(map(str, mlp_config))},{bwd_time},{bwd_energy}\n")

    profile_csv.close()
    print(f"Profile CSV saved to profile.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bayesian_profile_dir", default="/workspaces/Kareus/tests/bayesian", help="Directory containing BO results.")
    parser.add_argument("--prepost_profile_dir", default="/workspaces/Kareus/tests/fuser/prepost", help="Directory containing profiling results.")
    parser.add_argument("--tensor_parallel_size", default=2, type=int, help="Number of tensor-parallel ranks per stage. Times and energies are summed across these ranks.")
    parser.add_argument("--pipeline_parallel_size", default=2, type=int, help="Number of pipeline-parallel stages.")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
    parser.add_argument("--seq_len", default=4096, type=int, help="Sequence length.")
    parser.add_argument("--num_layers", default=28, type=int, help="Number of layers.")
    parser.add_argument("--p2p_power", default=90.0, type=float, help="GPU power consumption while blocking on P2P communication, in Watts.")
    args = parser.parse_args()

    main(
        args.bayesian_profile_dir,
        args.prepost_profile_dir,
        args.tensor_parallel_size,
        args.pipeline_parallel_size,
        args.batch_size,
        args.seq_len,
        args.num_layers,
        args.p2p_power,
    )