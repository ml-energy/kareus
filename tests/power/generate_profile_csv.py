"""Process a single run directory's timers to compute per-rank and per-pair averages.

Given a run directory containing a `timers/` subdirectory with files named like:

- `time-energy-<global_rank>-<local_rank>.csv` (monotonic time, cumulative energy)
- `instructions-<global_rank>-<local_rank>.csv` (instruction,start,end)

This script computes average time and energy per instruction for each global rank
and also for GPU pairs (0,1), (2,3), (4,5), (6,7).

Outputs two CSVs into the run directory:

- `profile_by_rank.csv` with header: rank,name,avg_time,avg_energy
- `profile_by_groups.csv` with header: group,name,avg_time,avg_energy
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class PiecewiseLinearModel:
    """An energy model that connects (x, y) measurements with straight lines.

    The model expects monotonically non-decreasing x and y.
    """

    def __init__(self, x_measurements: np.ndarray, y_measurements: np.ndarray) -> None:
        self.xs = x_measurements
        self.ys = y_measurements

        if len(self.xs) == 0 or len(self.ys) == 0:
            raise ValueError("Empty measurements for energy model.")
        if not np.all(np.diff(self.xs) >= 0):
            raise ValueError("X values must be sorted (non-decreasing).")
        if not np.all(np.diff(self.ys) >= 0):
            raise ValueError("Y values must be sorted (non-decreasing).")

    def __call__(self, x: float) -> float:
        if x < self.xs[0] or x > self.xs[-1]:
            raise ValueError(f"X value {x} out of range [{self.xs[0]}, {self.xs[-1]}].")
        return float(np.interp(x, self.xs, self.ys))


def _discover_rank_files(timers_dir: str) -> Tuple[Dict[int, str], Dict[int, str]]:
    """Return mappings from global_rank -> energy_path / instructions_path.

    Expects filenames like `time-energy-<global>-<local>.csv` and
    `instructions-<global>-<local>.csv`.
    """

    energy_map: Dict[int, str] = {}
    inst_map: Dict[int, str] = {}

    pattern_energy = re.compile(r"^time-energy-(\d+)-(\d+)\.csv$")
    pattern_inst = re.compile(r"^instructions-(\d+)-(\d+)\.csv$")

    for fname in os.listdir(timers_dir):
        full = os.path.join(timers_dir, fname)
        if not os.path.isfile(full):
            continue
        m_e = pattern_energy.match(fname)
        if m_e:
            g_rank = int(m_e.group(1))
            energy_map[g_rank] = full
            continue
        m_i = pattern_inst.match(fname)
        if m_i:
            g_rank = int(m_i.group(1))
            inst_map[g_rank] = full

    if not energy_map:
        raise RuntimeError(f"No energy files found in {timers_dir}.")
    if not inst_map:
        raise RuntimeError(f"No instruction files found in {timers_dir}.")

    missing = sorted(set(energy_map.keys()) ^ set(inst_map.keys()))
    if missing:
        raise RuntimeError(
            f"Mismatched ranks between energy and instructions; differing ranks: {missing}"
        )

    return energy_map, inst_map


def _compute_rank_averages(
    energy_path: str,
    instructions_path: str,
    num_microbatches: int,
    prof_iters: int,
    warmup_iters: int,
) -> Tuple[Dict[str, Tuple[float, float]], float | None, float | None, int]:
    """Compute avg time and energy per instruction for a rank.

    Returns mapping from instruction short name to tuple(avg_time, avg_energy).
    """

    energy_df = pd.read_csv(energy_path)
    model = PiecewiseLinearModel(energy_df.time.to_numpy(), energy_df.energy.to_numpy())

    inst_df = pd.read_csv(instructions_path)

    inst_name_map = {"forward-compute": "forward", "backward-compute": "backward"}

    results: Dict[str, Tuple[float, float]] = {}
    selected_starts: List[float] = []
    selected_ends: List[float] = []

    for inst, short in inst_name_map.items():
        sel = inst_df.query(f"instruction == '{inst}'")
        if sel.empty:
            continue

        inst_times: List[float] = []
        inst_energies: List[float] = []
        count = 0
        for _, row in sel.iterrows():
            # row: instruction, start, end
            count += 1
            if count <= warmup_iters * num_microbatches:
                continue
            if count > (prof_iters + warmup_iters) * num_microbatches:
                break
            start = float(row["start"])
            end = float(row["end"])
            inst_times.append(end - start)
            inst_energies.append(model(end) - model(start))
            selected_starts.append(start)
            selected_ends.append(end)

        if inst_times:
            expected = prof_iters * num_microbatches
            if not (len(inst_times) == expected and len(inst_energies) == expected):
                raise AssertionError(
                    f"Expected {expected} records for {inst}, got times={len(inst_times)} energies={len(inst_energies)}"
                )
            results[short] = (float(np.mean(inst_times)), float(np.mean(inst_energies)))

    # Determine profiling window and count energy samples inside it
    if selected_starts and selected_ends:
        profile_start = min(selected_starts)
        profile_end = max(selected_ends)
        times = energy_df.time.to_numpy()
        energy_samples_count = int(np.sum((times >= profile_start) & (times <= profile_end)))
    else:
        profile_start = None
        profile_end = None
        energy_samples_count = 0

    return results, profile_start, profile_end, energy_samples_count


def main(
    run_dir: str,
    num_microbatches: int,
    prof_iters: int,
    warmup_iters: int,
) -> None:
    timers_dir = os.path.join(run_dir, "timers")
    if not os.path.isdir(timers_dir):
        raise RuntimeError(f"timers directory not found: {timers_dir}")

    energy_map, inst_map = _discover_rank_files(timers_dir)
    ranks = sorted(energy_map.keys())

    # Per-rank results and debug energy-sample counts
    per_rank_results: Dict[int, Dict[str, Tuple[float, float]]] = {}
    per_rank_window: Dict[int, Tuple[float | None, float | None]] = {}
    per_rank_energy_counts: Dict[int, int] = {}
    rank_rows: List[Tuple[int, str, float, float]] = []
    for r in ranks:
        avgs, p_start, p_end, es_count = _compute_rank_averages(
            energy_map[r], inst_map[r], num_microbatches, prof_iters, warmup_iters
        )
        per_rank_results[r] = avgs
        per_rank_window[r] = (p_start, p_end)
        per_rank_energy_counts[r] = es_count
        if p_start is not None and p_end is not None:
            print(
                f"[DEBUG] Rank {r}: energy samples within profiling window = {es_count} "
                f"(window: [{p_start}, {p_end}])"
            )
        else:
            print(f"[DEBUG] Rank {r}: no profiling window found; energy samples = 0")
        for name, (avg_t, avg_e) in avgs.items():
            rank_rows.append((r, name, avg_t, avg_e))

    # Write per-rank CSV
    rank_csv_path = os.path.join(run_dir, "profile_by_rank.csv")
    with open(rank_csv_path, "w") as f:
        f.write("rank,name,avg_time,avg_energy\n")
        for r, name, avg_t, avg_e in rank_rows:
            f.write(f"{r},{name},{avg_t},{avg_e}\n")

    # Grouped pairs: (0,1), (2,3), (4,5), (6,7)
    # Use already computed per-rank results
    groups: List[Tuple[int, int]] = [(0, 1), (2, 3), (4, 5), (6, 7)]
    group_rows: List[Tuple[str, str, float, float]] = []
    for a, b in groups:
        if a not in per_rank_results or b not in per_rank_results:
            continue
        group_label = f"{a}-{b}"
        # For each instruction name present in either rank, average across available
        all_names = set(per_rank_results[a].keys()) | set(per_rank_results[b].keys())
        for name in sorted(all_names):
            vals: List[Tuple[float, float]] = []
            if name in per_rank_results[a]:
                vals.append(per_rank_results[a][name])
            if name in per_rank_results[b]:
                vals.append(per_rank_results[b][name])
            if not vals:
                continue
            avg_time = float(np.mean([v[0] for v in vals]))
            avg_energy = float(np.mean([v[1] for v in vals]))
            group_rows.append((group_label, name, avg_time, avg_energy))

    group_csv_path = os.path.join(run_dir, "profile_by_groups.csv")
    with open(group_csv_path, "w") as f:
        f.write("rank,name,avg_time,avg_energy\n")
        for idx, (grp, name, avg_t, avg_e) in enumerate(group_rows):
            r = idx // 2
            f.write(f"{r},{name},{avg_t},{avg_e}\n")

    print(f"Per-rank CSV saved to {rank_csv_path}")
    print(f"Per-group CSV saved to {group_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Path to a run directory that contains a 'timers/' subdirectory.",
    )
    parser.add_argument(
        "--num_microbatches", default=8, type=int, help="Number of microbatches."
    )
    parser.add_argument(
        "--num_prof_iters", default=10, type=int, help="Number of profiling iterations."
    )
    parser.add_argument(
        "--warmup_iters", default=5, type=int, help="Number of warmup iterations."
    )

    args = parser.parse_args()
    main(args.run_dir, args.num_microbatches, args.num_prof_iters, args.warmup_iters)


