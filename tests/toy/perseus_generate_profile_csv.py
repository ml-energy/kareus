"""Post-process time and energy profiling results into a profile CSV.

Scans a profiling directory (node0/, ...) where each node contains
per-frequency subdirectories with timer and energy CSVs.  For every
GPU frequency, it:

1. Builds a piecewise-linear energy model per rank from time-energy CSVs so
   that energy consumed during any time interval can be interpolated.
2. Reads per-rank instruction timing CSVs for forward and backward compute,
   skips warmup iterations, and computes per-rank average time and energy
   per instruction over the profiling window.
3. Groups ranks into pipeline stages (num_ranks / num_ranks_per_stage) and
   aggregates: max time across ranks in a stage (bottleneck), mean energy
   per rank in a stage.

Outputs a single profile.csv with columns:
    stage, instruction, frequency, time, energy

Usage:
    python generate_profile_csv.py --profile_dir <dir> [options]
"""

from __future__ import annotations

import argparse
from glob import glob
import os

import numpy as np
import pandas as pd


class PiecewiseLinearModel:
    """Energy model that connects (x, y) measurements with straight lines."""

    def __init__(self, x_measurements: np.ndarray, y_measurements: np.ndarray) -> None:
        self.xs = x_measurements
        self.ys = y_measurements

        if not np.all(np.diff(x_measurements) >= 0):
            raise ValueError("X values must be sorted.")
        if not np.all(np.diff(y_measurements) >= 0):
            raise ValueError("Y values must be sorted.")

    def __call__(self, x: float) -> float:
        if x < self.xs[0] or x > self.xs[-1]:
            raise ValueError(f"X value {x} is out of range [{self.xs[0]}, {self.xs[-1]}].")
        return np.interp(x, self.xs, self.ys).item()


def main(
    profile_dir: str,
    num_microbatches: int,
    prof_iters: int,
    warmup_iters: int,
    num_ranks_per_stage: int,
) -> None:
    print(f"Processing profiling results in {profile_dir}.")

    node_dirs = sorted([d for d in glob(f"{profile_dir}/node*") if os.path.isdir(d)])
    if not node_dirs:
        raise RuntimeError(
            f"No node directories found under {profile_dir}. "
            "Expected directories like node0, node1, ..."
        )

    first_node = node_dirs[0]
    freq_dirs_first_node = sorted(
        [d for d in glob(f"{first_node}/*") if os.path.isdir(d)],
        key=lambda p: int(os.path.basename(p)),
    )
    freq_names = [os.path.basename(d) for d in freq_dirs_first_node]
    if not freq_names:
        raise RuntimeError(f"No frequency directories found under {first_node}.")
    print(f"Frequencies: {[int(f) for f in freq_names]}")
    print(f"Found {len(freq_names)} frequency directories across nodes.")

    first_freq_name = freq_names[0]
    global_ranks_in_first_freq: set[int] = set()
    for node in node_dirs:
        energy_files = glob(f"{node}/{first_freq_name}/timers/time-energy-*.csv")
        for ef in energy_files:
            base = os.path.basename(ef)
            tok = base[:-4].rsplit("-", 2)
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
    print(f"Found {num_ranks} global ranks: {global_ranks_sorted}.")

    if num_ranks_per_stage <= 0:
        raise ValueError("num_ranks_per_stage must be a positive integer.")
    if num_ranks % num_ranks_per_stage != 0:
        raise RuntimeError(
            f"num_ranks ({num_ranks}) must be divisible by "
            f"num_ranks_per_stage ({num_ranks_per_stage})."
        )
    num_stages = num_ranks // num_ranks_per_stage
    print(f"Using num_ranks_per_stage={num_ranks_per_stage}; num_stages={num_stages}.")

    out_path = f"{profile_dir}/profile.csv"
    profile_csv = open(out_path, "w")
    profile_csv.write("stage,instruction,frequency,time,energy\n")

    for freq_name in freq_names:
        frequency = int(freq_name)
        print(f"Processing frequency {frequency} MHz.")

        models_by_g_rank: dict[int, PiecewiseLinearModel] = {}
        discovered_global_ranks: set[int] = set()
        for node in node_dirs:
            energy_files = glob(f"{node}/{freq_name}/timers/time-energy-*.csv")
            for ef in energy_files:
                base = os.path.basename(ef)
                tok = base[:-4].rsplit("-", 2)
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
                f"Expected {num_ranks} global energy files for frequency {frequency}, "
                f"but found {len(discovered_global_ranks)}."
            )
        models: list[PiecewiseLinearModel] = [models_by_g_rank[g] for g in global_ranks_sorted]

        timing_df_by_global_rank: dict[int, pd.DataFrame] = {}
        for node in node_dirs:
            timing_files = glob(f"{node}/{freq_name}/timers/instructions-*.csv")
            for tf in timing_files:
                base = os.path.basename(tf)
                tok = base[:-4].rsplit("-", 2)
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
                f"Expected {num_ranks} instruction timing results for frequency {frequency}, "
                f"but found {len(timing_df_by_global_rank)}."
            )
        timing_dfs = [timing_df_by_global_rank[g] for g in global_ranks_sorted]

        inst_name_map = {"forward-compute": "forward", "backward-compute": "backward"}

        stage_time_max: dict[tuple[int, str], float] = {}
        stage_energy_sums: dict[tuple[int, str], float] = {}

        for rank_index in range(num_ranks):
            print(f"  Processing rank {global_ranks_sorted[rank_index]} (index {rank_index}).")
            stage_idx = rank_index // num_ranks_per_stage
            for inst, name in inst_name_map.items():
                timing_df = timing_dfs[rank_index].query(f"instruction == '{inst}'")
                if timing_df.empty:
                    print(f"    No {inst} found.")
                    continue
                print(f"    Processing {inst}.")
                inst_times, inst_energies = [], []
                i = 0
                for _, (inst_val, start, end) in timing_df.iterrows():
                    i += 1
                    if i <= warmup_iters * num_microbatches:
                        continue
                    if i > (prof_iters + warmup_iters) * num_microbatches:
                        break
                    inst_times.append(end - start)
                    model = models[rank_index]
                    inst_energies.append(model(end) - model(start))

                if inst_times:
                    expected = prof_iters * num_microbatches
                    assert len(inst_times) == expected, (
                        f"Expected {expected} times, but got {len(inst_times)}."
                    )
                    assert len(inst_energies) == expected, (
                        f"Expected {expected} energies, but got {len(inst_energies)}."
                    )
                    avg_time = float(np.mean(inst_times))
                    avg_energy = float(np.mean(inst_energies))
                    stage_key = (stage_idx, name)
                    prev_max = stage_time_max.get(stage_key, 0.0)
                    stage_time_max[stage_key] = max(prev_max, avg_time)
                    stage_energy_sums[stage_key] = (
                        stage_energy_sums.get(stage_key, 0.0) + avg_energy
                    )

        for stage_idx in range(num_stages):
            for name in inst_name_map.values():
                stage_key = (stage_idx, name)
                stage_time = stage_time_max.get(stage_key, 0.0)
                stage_energy = stage_energy_sums.get(stage_key, 0.0) / num_ranks_per_stage
                profile_csv.write(
                    f"{stage_idx},{name},{frequency},{stage_time},{stage_energy}\n"
                )

    profile_csv.close()
    print(f"Profile CSV saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate profile.csv from profiling data."
    )
    parser.add_argument(
        "--profile_dir",
        required=True,
        help="Directory containing profiling results (with node0/ subdir).",
    )
    parser.add_argument("--num_microbatches", default=4, type=int)
    parser.add_argument("--num_prof_iters", default=10, type=int)
    parser.add_argument("--warmup_iters", default=5, type=int)
    parser.add_argument(
        "--num_ranks_per_stage", default=2, type=int,
        help="Number of ranks per pipeline stage (TP for this toy setup).",
    )
    args = parser.parse_args()

    main(
        args.profile_dir,
        args.num_microbatches,
        args.num_prof_iters,
        args.warmup_iters,
        args.num_ranks_per_stage,
    )
