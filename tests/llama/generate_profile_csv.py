"""Post-process time and energy profiling results from decoupled profiling mode."""

from __future__ import annotations

import argparse
import warnings
from glob import glob
from typing import Literal
import os

import numpy as np
import pandas as pd


class PiecewiseLinearModel:
    """A energy model that connects (x, y) measurements with a straight line."""
    
    def __init__(self, x_measurements: np.ndarray, y_measurements: np.ndarray) -> None:
        """Initialize the model with measurements."""
        self.xs = x_measurements
        self.ys = y_measurements

        # Both X and Y measurements must be sorted.
        if not np.all(np.diff(x_measurements) >= 0):
            raise ValueError("X values must be sorted.")
        if not np.all(np.diff(y_measurements) >= 0):
            raise ValueError("Y values must be sorted.")

    def __call__(self, x: float) -> float:
        """Return the estimated y value at the given x value."""
        if x < self.xs[0] or x > self.xs[-1]:
            raise ValueError(f"X value {x} is out of range [{self.xs[0]}, {self.xs[-1]}].")
        return np.interp(x, self.xs, self.ys).item()


def main(
    profile_dir: str,
    num_microbatches: int,
    prof_iters: int,
    warmup_iters: int,
    gpu_type: Literal["A100", "A40"],
    tensor_parallel_size: int,
) -> None:
    """Run the main routine."""
    print(f"Processing decoupled profiling results in {profile_dir}.")

    # Enumerate supported GPU frequencies.
    if gpu_type == "A100":
        freqs = np.arange(1410, 885, -30).tolist()
    elif gpu_type == "A40":
        freqs = np.arange(1740, 1300, -15).tolist()
    else:
        raise ValueError(f"Unsupported GPU type {gpu_type}.")
    print(f"Frequencies: {freqs}")

    # Find all frequency directories
    freq_dirs = sorted([d for d in glob(f"{profile_dir}/*") if os.path.isdir(d)])
    # if len(freq_dirs) != len(freqs):
    #     raise RuntimeError(f"Expected {len(freqs)} frequency directories, but found {len(freq_dirs)}.")
    print(f"Found {len(freq_dirs)} frequency directories.")

    # Determine number of ranks from the first frequency directory
    first_freq_dir = freq_dirs[0]
    energy_files = glob(f"{first_freq_dir}/timers/time-energy-*.csv")
    num_ranks = len(energy_files)
    if num_ranks == 0:
        raise RuntimeError("No energy polling results found in first frequency directory.")
    print(f"Found {num_ranks} ranks.")

    if tensor_parallel_size <= 0:
        raise ValueError("tensor_parallel_size must be a positive integer.")
    if num_ranks % tensor_parallel_size != 0:
        raise RuntimeError(
            f"num_ranks ({num_ranks}) must be divisible by tensor_parallel_size ({tensor_parallel_size})."
        )
    num_stages = num_ranks // tensor_parallel_size
    print(f"Using tensor_parallel_size={tensor_parallel_size}; num_stages={num_stages}.")

    # Process each frequency directory
    profile_csv = open(f"{profile_dir}/profile.csv", "w")
    profile_csv.write("stage,instruction,frequency,time,energy\n")
    
    for freq_idx, freq_dir in enumerate(freq_dirs):
        if freq_idx == len(freqs):
            break

        frequency = freqs[freq_idx]
        print(f"Processing frequency {frequency} Hz (directory: {freq_dir}).")
        
        # Read in energy polling results for this frequency
        energy_files = glob(f"{freq_dir}/timers/time-energy-*.csv")
        if len(energy_files) != num_ranks:
            raise RuntimeError(f"Expected {num_ranks} energy files in {freq_dir}, but found {len(energy_files)}.")
        
        models: list[PiecewiseLinearModel] = []
        for rank in range(num_ranks):
            df = pd.read_csv(f"{freq_dir}/timers/time-energy-{rank}.csv")
            model = PiecewiseLinearModel(df.time.to_numpy(), df.energy.to_numpy())
            models.append(model)
            del df

        # Read in instruction timing results for this frequency
        timing_files = sorted(glob(f"{freq_dir}/timers/instructions-*.csv"))
        if len(timing_files) != num_ranks:
            raise RuntimeError(
                f"Expected {num_ranks} instruction timing results in {freq_dir}, but found {len(timing_files)}."
            )
        timing_dfs = [pd.read_csv(f) for f in timing_files]

        # # Only choose odd index "batch_input" records in the last rank.
        # # That's because for each forward pass in the last rank, two "batch_input"s are
        # # recorded: one from recv_grad_send_activationa and one from actual load_microbatch.
        # last_rank_df = timing_dfs[-1]
        # if "batch_input" in last_rank_df.instruction.values:
        #     other_records = last_rank_df.query("instruction != 'batch_input'")
        #     batch_input_records = (
        #         last_rank_df
        #         .query("instruction == 'batch_input'")
        #         .reset_index(drop=True)
        #         .iloc[1::2]
        #     )
        #     if (batch_input_records.end - batch_input_records.start).min() < 0.0001:
        #         warnings.warn(
        #             "Last rank batch_input records after filtering includes records shorter than 0.1 ms."
        #         )
        #     timing_dfs[-1] = pd.concat([other_records, batch_input_records]).reset_index(drop=True)

        # # Assert same number of records.
        # if "batch_input" in timing_dfs[0].instruction.values:
        #     lens = [len(df) for df in timing_dfs]
        #     for rank in range(num_ranks - 1):
        #         if lens[rank] != lens[rank + 1]:
        #             raise ValueError(
        #                 f"Rank {rank} has {lens[rank]} records, but rank {rank + 1} has {lens[rank + 1]} records."
        #             )
        
        # For each rank, the timing dataframe contains instruction start and end
        # timing measurements.
        inst_name_map = {"forward-compute": "forward", "backward-compute": "backward"}

        # Accumulate per-stage sums over tensor_parallel_size ranks
        stage_time_sums: dict[tuple[int, str], float] = {}
        stage_energy_sums: dict[tuple[int, str], float] = {}

        for rank in range(num_ranks):
            print(f"  Processing rank {rank}.")
            stage_idx = rank // tensor_parallel_size
            for inst, name in inst_name_map.items():
                timing_df = timing_dfs[rank].query(f"instruction == '{inst}'")
                if timing_df.empty:
                    print(f"    No {inst} found.")
                    # Treat as zero contribution
                    continue
                print(f"    Processing {inst}.")
                inst_times, inst_energies = [], []
                i = 0
                for _, (inst, start, end) in timing_df.iterrows():
                    i += 1
                    if i <= warmup_iters * num_microbatches:
                        continue
                    if i > (prof_iters + warmup_iters) * num_microbatches:
                        break
                    inst_times.append(end - start)
                    model = models[rank]
                    inst_energies.append(model(end) - model(start))

                # Calculate average time and energy for this instruction at this frequency
                if inst_times:  # Only contribute if we have data after warmup
                    expected = prof_iters * num_microbatches
                    assert len(inst_times) == expected, f"Expected {expected} times, but got {len(inst_times)}."
                    assert len(inst_energies) == expected, f"Expected {expected} energies, but got {len(inst_energies)}."
                    avg_time = float(np.mean(inst_times))
                    avg_energy = float(np.mean(inst_energies))
                    stage_key = (stage_idx, name)
                    stage_time_sums[stage_key] = stage_time_sums.get(stage_key, 0.0) + avg_time
                    stage_energy_sums[stage_key] = stage_energy_sums.get(stage_key, 0.0) + avg_energy

        # After processing all ranks, write per-stage aggregated rows
        for stage_idx in range(num_stages):
            for name in inst_name_map.values():
                stage_key = (stage_idx, name)
                avg_time = stage_time_sums.get(stage_key, 0.0) / tensor_parallel_size
                avg_energy = stage_energy_sums.get(stage_key, 0.0) / tensor_parallel_size
                profile_csv.write(f"{stage_idx},{name},{frequency},{avg_time},{avg_energy}\n")
    
    profile_csv.close()
    print(f"Profile CSV saved to {profile_dir}/profile.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_dir", default="nemo_experiments/megatron_llama_3_2_3b", help="Directory containing profiling results.")
    parser.add_argument("--num_microbatches", default=8, type=int, help="Number of microbatches.")
    parser.add_argument("--num_prof_iters", default=100, type=int, help="Number of profiling iterations.")
    parser.add_argument("--warmup_iters", default=10, type=int, help="Number of warmup iterations.")
    parser.add_argument("--gpu_type", default="A100", choices=["A40", "A100"], help="Name of the GPU type.")
    parser.add_argument("--tensor_parallel_size", default=2, type=int, help="Number of tensor-parallel ranks per stage. Times and energies are summed across these ranks.")
    args = parser.parse_args()

    main(args.profile_dir, args.num_microbatches, args.num_prof_iters, args.warmup_iters, args.gpu_type, args.tensor_parallel_size)