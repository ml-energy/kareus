"""Parse baseline time and energy results from zeus monitor files."""

from __future__ import annotations

import argparse
import os
from glob import glob
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def parse_results(
    path: str,
    num_ranks: int = None,
    warmup_iters: int = 10,
    profile_iters: int = 100,
) -> Dict[int, Tuple[float, float]]:
    """
    Parse baseline time and energy results from zeus monitor files.
    
    Args:
        path: Path to directory containing zeus_monitor_localrank-*.txt files
        num_ranks: Number of ranks to process. If None, auto-detect from files.
        warmup_iters: Number of warmup iterations to skip
        profile_iters: Number of profiling iterations to use for averaging
    
    Returns:
        Dictionary mapping rank to (avg_time, avg_energy) tuple
    """
    
    # Auto-detect number of ranks if not specified
    if num_ranks is None:
        files = glob(f"{path}/zeus_monitor_localrank-*.txt")
        if not files:
            raise RuntimeError(f"No files found in {path}")
        num_ranks = len(files)
    
    
    results = {}
    for rank in range(num_ranks):
        file = f"{path}/zeus_monitor_localrank-{rank}.txt"
        if not os.path.exists(file):
            print(f"  Warning: file for rank {rank} not found: {file}")
            continue
            
        df = pd.read_csv(file)
        training_steps = df[df['window_name'] == 'training_step']
        if training_steps.empty:
            print(f"    No training_step entries found for rank {rank}")
            continue

        if len(training_steps) < warmup_iters + profile_iters:
            print(f"    Warning: Not enough entries ({len(training_steps)}) for warmup ({warmup_iters}) + profile ({profile_iters}) iterations")
            return {}
        else:
            # Skip warmup iterations and take exactly profile_iters entries
            selected_steps = training_steps.iloc[warmup_iters:warmup_iters + profile_iters]
            
        times = selected_steps['elapsed_time'].values
        energies = selected_steps[f'gpu{rank}_energy'].values
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
        
        total_time = sum([x[0] for x in results.values()])
        total_energy = sum([x[1] for x in results.values()])
        f.write(f"total,{total_time},{total_energy}\n")


def main() -> None:
    """Main function to parse baseline results and save to CSV."""
    parser = argparse.ArgumentParser(description="Parse baseline time and energy results from zeus monitor files")
    parser.add_argument(
        "--baseline_path", 
        default="/workspaces/Kareus/tests/perseus/nemo_experiments/megatron_llama_3_2_1b/baseline",
        help="Path to directory containing zeus_monitor_localrank-*.txt files"
    )
    parser.add_argument(
        "--perseus_path", 
        default="/workspaces/Kareus/tests/perseus/nemo_experiments/megatron_llama_3_2_1b/optimized",
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
        default=100,
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
    
    print("Baseline results:")
    for rank, (avg_time, avg_energy) in baseline_results.items():
        print(f"  Rank {rank}: {avg_time:.6f}s, {avg_energy:.6f}J")
    print(f"  Total: {sum([x[0] for x in baseline_results.values()]):.6f}s, {sum([x[1] for x in baseline_results.values()]):.6f}J")

    print("Perseus results:")
    for rank, (avg_time, avg_energy) in perseus_results.items():
        print(f"  Rank {rank}: {avg_time:.6f}s, {avg_energy:.6f}J")
    print(f"  Total: {sum([x[0] for x in perseus_results.values()]):.6f}s, {sum([x[1] for x in perseus_results.values()]):.6f}J")
            


if __name__ == "__main__":
    main() 