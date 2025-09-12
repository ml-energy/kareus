"""Post-process time and energy profiling results from decoupled profiling mode."""

from __future__ import annotations

import argparse
import warnings
from glob import glob
from typing import Literal
import os

import numpy as np
import pandas as pd

# Import shared defaults
import sys
FUSER_DIR = os.path.join(os.path.dirname(__file__), '..', 'fuser')
if FUSER_DIR not in sys.path:
    sys.path.append(FUSER_DIR)
from common_config import FuserTestConfig  # noqa: E402


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

    # Discover nodes
    node_dirs = sorted([d for d in glob(f"{profile_dir}/node*") if os.path.isdir(d)])
    if not node_dirs:
        raise RuntimeError(f"No node directories found under {profile_dir}. Expected directories like node0, node1, ...")

    # Discover frequency directories (names must be exact frequency numbers) from the first node
    first_node = node_dirs[0]
    freq_dirs_first_node = sorted([d for d in glob(f"{first_node}/*") if os.path.isdir(d)], key=lambda p: int(os.path.basename(p)))
    freq_names = [os.path.basename(d) for d in freq_dirs_first_node]
    if not freq_names:
        raise RuntimeError(f"No frequency directories found under {first_node}.")
    print(f"Frequencies: {[int(f) for f in freq_names]}")
    print(f"Found {len(freq_names)} frequency directories across nodes.")

    # Determine number of GLOBAL ranks from the first frequency (union across all nodes)
    first_freq_name = freq_names[0]
    global_ranks_in_first_freq: set[int] = set()
    for node in node_dirs:
        energy_files = glob(f"{node}/{first_freq_name}/timers/time-energy-*.csv")
        for ef in energy_files:
            base = os.path.basename(ef)
            tok = base[:-4].rsplit('-', 2)  # [prefix, global, local]
            if len(tok) != 3:
                continue
            try:
                g_rank = int(tok[-2])
            except ValueError:
                continue
            global_ranks_in_first_freq.add(g_rank)
    num_ranks = len(global_ranks_in_first_freq)
    if num_ranks == 0:
        raise RuntimeError("No energy polling results found in first frequency across nodes.")
    global_ranks_sorted: list[int] = sorted(global_ranks_in_first_freq)
    global_rank_to_index: dict[int, int] = {g: i for i, g in enumerate(global_ranks_sorted)}
    print(f"Found {num_ranks} global ranks: {global_ranks_sorted}.")

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
    
    for freq_name in freq_names:
        frequency = int(freq_name)
        print(f"Processing frequency {frequency} Hz.")
        
        # Build PiecewiseLinearModel per GLOBAL rank from all nodes
        models_by_g_rank: dict[int, PiecewiseLinearModel] = {}
        discovered_global_ranks: set[int] = set()
        for node in node_dirs:
            energy_files = glob(f"{node}/{freq_name}/timers/time-energy-*.csv")
            for ef in energy_files:
                base = os.path.basename(ef)
                tok = base[:-4].rsplit('-', 2)
                if len(tok) != 3:
                    continue
                try:
                    g_rank = int(tok[-2])
                except ValueError:
                    continue
                if g_rank in discovered_global_ranks:
                    continue
                df = pd.read_csv(ef)
                model = PiecewiseLinearModel(df.time.to_numpy(), df.energy.to_numpy())
                models_by_g_rank[g_rank] = model
                discovered_global_ranks.add(g_rank)
                del df
        if len(discovered_global_ranks) != num_ranks:
            raise RuntimeError(
                f"Expected {num_ranks} global energy files for frequency {frequency}, but found {len(discovered_global_ranks)}."
            )
        models: list[PiecewiseLinearModel] = [models_by_g_rank[g] for g in global_ranks_sorted]

        # Read in instruction timing results for this frequency
        timing_df_by_global_rank: dict[int, pd.DataFrame] = {}
        for node in node_dirs:
            timing_files = glob(f"{node}/{freq_name}/timers/instructions-*.csv")
            for tf in timing_files:
                base = os.path.basename(tf)
                tok = base[:-4].rsplit('-', 2)
                if len(tok) != 3:
                    continue
                try:
                    g_rank = int(tok[-2])
                except ValueError:
                    continue
                if g_rank in timing_df_by_global_rank:
                    continue
                timing_df_by_global_rank[g_rank] = pd.read_csv(tf)
        if len(timing_df_by_global_rank) != num_ranks:
            raise RuntimeError(
                f"Expected {num_ranks} instruction timing results for frequency {frequency}, but found {len(timing_df_by_global_rank)}."
            )
        timing_dfs = [timing_df_by_global_rank[g] for g in global_ranks_sorted]

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

        # Accumulate per-stage aggregates over tensor_parallel_size ranks
        # Time: use max across ranks; Energy: sum across ranks
        stage_time_max: dict[tuple[int, str], float] = {}
        stage_energy_sums: dict[tuple[int, str], float] = {}

        for rank_index in range(num_ranks):
            print(f"  Processing rank {global_ranks_sorted[rank_index]} (index {rank_index}).")
            stage_idx = rank_index // tensor_parallel_size
            for inst, name in inst_name_map.items():
                timing_df = timing_dfs[rank_index].query(f"instruction == '{inst}'")
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
                    model = models[rank_index]
                    inst_energies.append(model(end) - model(start))

                # Calculate average time and energy for this instruction at this frequency
                if inst_times:  # Only contribute if we have data after warmup
                    expected = prof_iters * num_microbatches
                    assert len(inst_times) == expected, f"Expected {expected} times, but got {len(inst_times)}."
                    assert len(inst_energies) == expected, f"Expected {expected} energies, but got {len(inst_energies)}."
                    avg_time = float(np.mean(inst_times))
                    avg_energy = float(np.mean(inst_energies))
                    stage_key = (stage_idx, name)
                    prev_max = stage_time_max.get(stage_key, 0.0)
                    stage_time_max[stage_key] = max(prev_max, avg_time)
                    stage_energy_sums[stage_key] = stage_energy_sums.get(stage_key, 0.0) + avg_energy

        # After processing all ranks, write per-stage aggregated rows
        for stage_idx in range(num_stages):
            for name in inst_name_map.values():
                stage_key = (stage_idx, name)
                stage_time = stage_time_max.get(stage_key, 0.0)
                stage_energy = stage_energy_sums.get(stage_key, 0.0)
                profile_csv.write(f"{stage_idx},{name},{frequency},{stage_time},{stage_energy}\n")
    
    profile_csv.close()
    print(f"Profile CSV saved to {profile_dir}/profile.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_dir", default="nemo_experiments/megatron_llama_3_2_3b/", help="Directory containing profiling results.")
    parser.add_argument("--num_microbatches", default=FuserTestConfig.DEFAULT_NUM_MICROBATCHES, type=int, help="Number of microbatches.")
    parser.add_argument("--num_prof_iters", default=20, type=int, help="Number of profiling iterations.")
    parser.add_argument("--warmup_iters", default=10, type=int, help="Number of warmup iterations.")
    parser.add_argument("--gpu_type", default="A100", choices=["A40", "A100"], help="Name of the GPU type.")
    parser.add_argument("--tensor_parallel_size", default=FuserTestConfig.DEFAULT_WORLD_SIZE, type=int, help="Number of tensor-parallel ranks per stage. Times and energies are summed across these ranks.")
    args = parser.parse_args()

    main(args.profile_dir, args.num_microbatches, args.num_prof_iters, args.warmup_iters, args.gpu_type, args.tensor_parallel_size)