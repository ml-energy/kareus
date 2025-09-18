"""Parse baseline time and energy results from zeus monitor files."""

from __future__ import annotations

import argparse
import os
import re
from glob import glob
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_results(
    path: str,
    num_ranks: int = None,
    warmup_iters: int = 10,
    profile_iters: int = 20,
) -> Dict[int, Tuple[float, float]]:
    """
    Parse baseline time and energy results from zeus monitor files.
    
    Args:
        path: Path to directory containing monitor files matching pattern:
              - zeus_monitor_global_rank-<global_rank>_local_rank-<local_rank>.txt
        num_ranks: Number of ranks to process. If None, auto-detect from files.
        warmup_iters: Number of warmup iterations to skip
        profile_iters: Number of profiling iterations to use for averaging
    
    Returns:
        Dictionary mapping rank to (avg_time, avg_energy) tuple
    """
    
    # Discover files matching the new naming pattern only
    files = glob(f"{path}/zeus_monitor_global_rank-*_local_rank-*.txt")
    if not files:
        raise RuntimeError(f"No files found in {path}")

    # Build a mapping from global_rank to (file path, local_rank)
    global_to_file: Dict[int, Tuple[str, int]] = {}
    for file_path in files:
        basename = os.path.basename(file_path)
        m_new = re.match(r"zeus_monitor_global_rank-(\d+)_local_rank-(\d+)\.txt$", basename)
        if m_new:
            # Use global rank for keying results; keep local rank to pick energy column
            global_rank = int(m_new.group(1))
            local_rank = int(m_new.group(2))
            global_to_file[global_rank] = (file_path, local_rank)
            continue

    if not global_to_file:
        raise RuntimeError(f"No valid zeus monitor files matched expected patterns in {path}")
    
    
    results = {}

    # Determine which global ranks to process
    if num_ranks is None:
        target_ranks = sorted(global_to_file.keys())
    else:
        target_ranks = list(range(num_ranks))

    for rank in target_ranks:
        file_and_local = global_to_file.get(rank)
        if not file_and_local or not os.path.exists(file_and_local[0]):
            print(f"  Warning: file for global rank {rank} not found")
            continue
            
        file = file_and_local[0]
        local_rank = file_and_local[1]
        df = pd.read_csv(file)
        training_steps = df[df['window_name'] == 'training_step_fwd_bwd_step_call']
        if training_steps.empty:
            print(f"    No training_step entries found for global rank {rank}")
            continue

        if len(training_steps) < warmup_iters + profile_iters:
            print(f"    Warning: Not enough entries ({len(training_steps)}) for warmup ({warmup_iters}) + profile ({profile_iters}) iterations")
            return {}
        else:
            # Skip warmup iterations and take exactly profile_iters entries
            selected_steps = training_steps.iloc[warmup_iters:warmup_iters + profile_iters]
            
        times = selected_steps['elapsed_time'].values
        energies = selected_steps[f'gpu{local_rank}_energy'].values
        avg_time = np.mean(times)
        avg_energy = np.mean(energies)
        
        results[rank] = (avg_time, avg_energy)
    
    return results


def save_csv(
    results: Dict[int, Tuple[float, float]], 
    output_file: str = "baseline_results.csv"
) -> None:
    """
    Save results to a CSV file.
    
    Args:
        results: Dictionary mapping rank to (avg_time, avg_energy) tuple
        output_file: Output CSV file path
    """
    print(f"Saving results to {output_file}")
    
    with open(output_file, "w") as f:
        f.write("rank,avg_time,avg_energy\n")
        for rank, (avg_time, avg_energy) in results.items():
            f.write(f"{rank},{avg_time},{avg_energy}\n")
        
        max_time = max([x[0] for x in results.values()]) if results else 0.0
        total_energy = sum([x[1] for x in results.values()])
        f.write(f"total,{max_time},{total_energy}\n")


def aggregate_time_energy(results: Dict[int, Tuple[float, float]]) -> Tuple[float, float]:
    """Aggregate per-rank results into a single (time, energy) point.

    Uses max time across ranks and sum of energies across ranks.
    Returns (0.0, 0.0) if results is empty.
    """
    if not results:
        return 0.0, 0.0
    max_time = max([x[0] for x in results.values()])
    total_energy = sum([x[1] for x in results.values()])
    return max_time, total_energy


def plot_time_energy(
    baseline_results: Dict[int, Tuple[float, float]],
    perseus_results: Dict[int, Tuple[float, float]],
    nanobatch_results: Dict[int, Tuple[float, float]],
    nanobatch_op_results: Dict[int, Tuple[float, float]],
    kareus_results: Dict[int, Tuple[float, float]],
    output_file: str = "time_energy_scatter.png",
) -> None:
    """Create a scatter plot of aggregated time vs energy for four runs."""
    labels = ["Baseline", "Perseus", "Nanobatch", "Nanobatch+Perseus", "Kareus"]
    dicts = [baseline_results, perseus_results, nanobatch_results, nanobatch_op_results, kareus_results]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    xs, ys, used_labels, used_colors = [], [], [], []
    for label, res, color in zip(labels, dicts, colors):
        if res is None:
            continue
        x, y = aggregate_time_energy(res)
        # Skip truly empty
        if x == 0.0 and y == 0.0:
            continue
        xs.append(x)
        ys.append(y)
        used_labels.append(label)
        used_colors.append(color)

    if not xs:
        print("[!] No data to plot for time-energy scatter; skipping figure.")
        return

    plt.figure(figsize=(7, 5))
    for x, y, label, color in zip(xs, ys, used_labels, used_colors):
        plt.scatter(x, y, s=90, alpha=0.85, label=label, color=color)
        plt.text(x, y, f"  {label}", fontsize=10, va="center", ha="left")

    plt.xlabel("time (s)")
    plt.ylabel("total energy (J)")
    plt.title("Time vs Energy across runs")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()
    print(f"[\u2713] Figure written to {output_file}")


def main() -> None:
    """Main function to parse baseline results and save to CSV."""
    parser = argparse.ArgumentParser(description="Parse baseline time and energy results from zeus monitor files")
    parser.add_argument(
        "--baseline_path", 
        default="nemo_experiments/megatron_llama_3_2_1b/baseline",
        help="Path to directory containing zeus_monitor_localrank-*.txt files"
    )
    parser.add_argument(
        "--perseus_path", 
        default="nemo_experiments/megatron_llama_3_2_1b/optimized",
        help="Path to directory containing zeus_monitor_localrank-*.txt files"
    )
    parser.add_argument(
        "--nanobatch_path", 
        default="../kareus/nemo_experiments/megatron_llama_3_2_1b/nanobatch_baseline",
        help="Path to directory containing zeus_monitor_localrank-*.txt files"
    )
    parser.add_argument(
        "--nanobatch_op_path", 
        default="../kareus/nemo_experiments/megatron_llama_3_2_1b/nanobatch_perseus",
        help="Path to directory containing zeus_monitor_localrank-*.txt files"
    )
    parser.add_argument(
        "--kareus_path", 
        default="../kareus/nemo_experiments/megatron_llama_3_2_1b/kareus",
        help="Path to directory containing zeus_monitor_localrank-*.txt files"
    )
    parser.add_argument(
        "--num_ranks", 
        type=int, 
        default=None,
        help="Number of ranks to process. If not specified, auto-detect from files."
    )
    parser.add_argument(
        "--warmup_iters", 
        type=int, 
        default=10,
        help="Number of warmup iterations to skip"
    )
    parser.add_argument(
        "--profile_iters", 
        type=int, 
        default=20,
        help="Number of profiling iterations to use for averaging"
    )
    
    args = parser.parse_args()
    
    baseline_results = parse_results(
        args.baseline_path, 
        args.num_ranks,
        args.warmup_iters,
        args.profile_iters
    )
    save_csv(baseline_results, "baseline_results.csv")

    perseus_results = parse_results(
        args.perseus_path, 
        args.num_ranks,
        args.warmup_iters,
        args.profile_iters
    )
    save_csv(perseus_results, "perseus_results.csv")
    
    nanobatch_results = parse_results(
        args.nanobatch_path, 
        args.num_ranks,
        args.warmup_iters,
        args.profile_iters
    )
    save_csv(nanobatch_results, "nanobatch_results.csv")
    
    nanobatch_op_results = parse_results(
        args.nanobatch_op_path, 
        args.num_ranks,
        args.warmup_iters,
        args.profile_iters
    )
    save_csv(nanobatch_op_results, "nanobatch_op_results.csv")
    
    kareus_results = parse_results(
        args.kareus_path, 
        args.num_ranks,
        args.warmup_iters,
        args.profile_iters
    )
    save_csv(kareus_results, "kareus_results.csv")
    
    # Plot aggregated time-energy points for the four runs
    plot_time_energy(
        baseline_results,
        perseus_results,
        nanobatch_results,
        nanobatch_op_results,
        kareus_results,
        output_file="time_energy_scatter.png",
    )
    
    print("Baseline results:")
    for rank, (avg_time, avg_energy) in baseline_results.items():
        print(f"  Rank {rank}: {avg_time:.6f}s, {avg_energy:.6f}J")
    baseline_max_time = max([x[0] for x in baseline_results.values()]) if baseline_results else 0.0
    baseline_sum_energy = sum([x[1] for x in baseline_results.values()])
    print(f"  Total: {baseline_max_time:.6f}s, {baseline_sum_energy:.6f}J")

    print("Perseus results:")
    for rank, (avg_time, avg_energy) in perseus_results.items():
        print(f"  Rank {rank}: {avg_time:.6f}s, {avg_energy:.6f}J")
    perseus_max_time = max([x[0] for x in perseus_results.values()]) if perseus_results else 0.0
    perseus_sum_energy = sum([x[1] for x in perseus_results.values()])
    print(f"  Total: {perseus_max_time:.6f}s, {perseus_sum_energy:.6f}J")
    
    print("Nanobatch results:")
    for rank, (avg_time, avg_energy) in nanobatch_results.items():
        print(f"  Rank {rank}: {avg_time:.6f}s, {avg_energy:.6f}J")
    nanobatch_max_time = max([x[0] for x in nanobatch_results.values()]) if nanobatch_results else 0.0
    nanobatch_sum_energy = sum([x[1] for x in nanobatch_results.values()])
    print(f"  Total: {nanobatch_max_time:.6f}s, {nanobatch_sum_energy:.6f}J")

    print("Nanobatch+Perseus results:")
    for rank, (avg_time, avg_energy) in nanobatch_op_results.items():
        print(f"  Rank {rank}: {avg_time:.6f}s, {avg_energy:.6f}J")
    nanobatch_op_max_time = max([x[0] for x in nanobatch_op_results.values()]) if nanobatch_op_results else 0.0
    nanobatch_op_sum_energy = sum([x[1] for x in nanobatch_op_results.values()])
    print(f"  Total: {nanobatch_op_max_time:.6f}s, {nanobatch_op_sum_energy:.6f}J")
    
    print("Kareus results:")
    for rank, (avg_time, avg_energy) in kareus_results.items():
        print(f"  Rank {rank}: {avg_time:.6f}s, {avg_energy:.6f}J")
    kareus_max_time = max([x[0] for x in kareus_results.values()]) if kareus_results else 0.0
    kareus_sum_energy = sum([x[1] for x in kareus_results.values()])
    print(f"  Total: {kareus_max_time:.6f}s, {kareus_sum_energy:.6f}J")
            


if __name__ == "__main__":
    main() 