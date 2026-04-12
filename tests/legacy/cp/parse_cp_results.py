from __future__ import annotations

import argparse
import os
import re
from glob import glob
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd


def compute_avg_or_last(
    training_steps: pd.DataFrame,
    local_rank: int,
    warmup_iters: int,
    profile_iters: int,
) -> Tuple[float, float]:
    """Return (avg_time_s, avg_energy_J) for a rank.

    If there are at least warmup_iters + profile_iters rows, average the
    slice [warmup_iters : warmup_iters + profile_iters]. Otherwise, use the
    last available iteration as the result.
    """
    energy_col = f"gpu{local_rank}_energy"
    if energy_col not in training_steps.columns:
        # Fallback: look for an exact match anyway (kept for symmetry with callers)
        alt_cols = [c for c in training_steps.columns if re.fullmatch(rf"gpu{local_rank}_energy", c)]
        if alt_cols:
            energy_col = alt_cols[0]

    if len(training_steps) >= warmup_iters + profile_iters:
        selected = training_steps.iloc[warmup_iters : warmup_iters + profile_iters]
        times = selected["elapsed_time"].values
        energies = selected[energy_col].values if energy_col in selected.columns else None
        avg_time = float(np.mean(times))
        avg_energy = float(np.mean(energies)) if energies is not None else 0.0
        return avg_time, avg_energy

    # Not enough iterations: use last row
    last = training_steps.iloc[-1]
    last_time = float(last["elapsed_time"]) if "elapsed_time" in last else 0.0
    last_energy = float(last[energy_col]) if energy_col in last else 0.0
    return last_time, last_energy


def parse_run_dir(
    path: str,
    warmup_iters: int = 5,
    profile_iters: int = 10,
) -> Dict[int, Tuple[float, float]]:
    """Parse per-rank averages for a single run directory.

    Returns mapping of global rank -> (avg_time_s, avg_energy_J)
    using window_name == 'training_step_fwd_bwd_step_call'.
    """
    files = glob(os.path.join(path, "zeus_monitor_global_rank-*_local_rank-*.txt"))
    if not files:
        return {}

    global_to_file: Dict[int, Tuple[str, int]] = {}
    for file_path in files:
        basename = os.path.basename(file_path)
        m = re.match(r"zeus_monitor_global_rank-(\d+)_local_rank-(\d+)\.txt$", basename)
        if not m:
            continue
        global_rank = int(m.group(1))
        local_rank = int(m.group(2))
        global_to_file[global_rank] = (file_path, local_rank)

    if not global_to_file:
        return {}

    results: Dict[int, Tuple[float, float]] = {}
    for rank, (file_path, local_rank) in sorted(global_to_file.items()):
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        training_steps = df[df["window_name"] == "training_step_fwd_bwd_step_call"]
        if training_steps.empty:
            continue

        avg_time, avg_energy = compute_avg_or_last(
            training_steps, local_rank, warmup_iters, profile_iters
        )
        results[rank] = (avg_time, avg_energy)

    return results


def aggregate_time_energy(per_rank: Dict[int, Tuple[float, float]]) -> Tuple[float, float]:
    """Aggregate per-rank averages into a single (time, energy) point.

    - time: max across ranks
    - energy: sum across ranks
    Returns (0.0, 0.0) if empty.
    """
    if not per_rank:
        return 0.0, 0.0
    times = [v[0] for v in per_rank.values()]
    energies = [v[1] for v in per_rank.values()]
    return float(max(times)), float(sum(energies))


def find_model_dirs(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    entries = [os.path.join(root, d) for d in os.listdir(root)]
    return [d for d in entries if os.path.isdir(d)]


def find_run_dirs(model_dir: str) -> List[str]:
    """Return subdirectories that look like run dirs (contain zeus monitor files)."""
    run_dirs: List[str] = []
    for name in os.listdir(model_dir):
        path = os.path.join(model_dir, name)
        if not os.path.isdir(path):
            continue
        # Skip timestamp folders unless they contain monitor files directly
        files = glob(os.path.join(path, "zeus_monitor_global_rank-*_local_rank-*.txt"))
        if files:
            run_dirs.append(path)
    return run_dirs


def classify_run_dir(run_dir: str) -> Optional[Tuple[str, str]]:
    """Return (flavor, config_key) where flavor in {"megatron", "kareus"}.

    Config key is the suffix after the first underscore, e.g. 'tp8_16_8k'.
    """
    base = os.path.basename(run_dir)
    if base.startswith("megatron_"):
        return "megatron", base.split("_", 1)[1]
    if base.startswith("kareus_"):
        return "kareus", base.split("_", 1)[1]
    if base.startswith("nanobatch_"):
        return "nanobatch", base.split("_", 1)[1]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Megatron vs Kareus time/energy across configs")
    parser.add_argument(
        "--root",
        default="nemo_experiments",
        help="Root directory containing model subdirectories",
    )
    parser.add_argument(
        "--warmup_iters", type=int, default=5, help="Warmup iterations to skip"
    )
    parser.add_argument(
        "--profile_iters", type=int, default=10, help="Iterations to average after warmup"
    )
    parser.add_argument(
        "--output_csv",
        default="time_energy_summary.csv",
        help="Path to write aggregated CSV",
    )
    args = parser.parse_args()

    rows: List[Dict[str, object]] = []

    for model_dir in find_model_dirs(args.root):
        model_name = os.path.basename(model_dir)
        if model_name.startswith("megatron_"):
            model_name = model_name[len("megatron_") :]
        runs = find_run_dirs(model_dir)

        # Collect results by flavor/config
        megatron_map: Dict[str, Tuple[float, float]] = {}
        kareus_map: Dict[str, Tuple[float, float]] = {}
        nanobatch_map: Dict[str, Tuple[float, float]] = {}

        for run_dir in runs:
            classified = classify_run_dir(run_dir)
            if classified is None:
                continue
            flavor, config_key = classified
            per_rank = parse_run_dir(run_dir, args.warmup_iters, args.profile_iters)
            agg_time, agg_energy = aggregate_time_energy(per_rank)
            if agg_time == 0.0 and agg_energy == 0.0:
                continue
            if flavor == "megatron":
                megatron_map[config_key] = (agg_time, agg_energy)
            elif flavor == "kareus":
                kareus_map[config_key] = (agg_time, agg_energy)
            elif flavor == "nanobatch":
                nanobatch_map[config_key] = (agg_time, agg_energy)

        # Report one row per config present in Megatron; others may be empty
        for config_key in sorted(megatron_map.keys()):
            m_time, m_energy = megatron_map[config_key]
            k_vals = kareus_map.get(config_key)
            n_vals = nanobatch_map.get(config_key)

            k_time = round(k_vals[0], 6) if k_vals else None
            k_energy = round(k_vals[1], 6) if k_vals else None
            n_time = round(n_vals[0], 6) if n_vals else None
            n_energy = round(n_vals[1], 6) if n_vals else None

            # Percentage reduction relative to Megatron
            k_time_red = round(100.0 * (m_time - k_vals[0]) / m_time, 2) if k_vals and m_time else None
            k_energy_red = round(100.0 * (m_energy - k_vals[1]) / m_energy, 2) if k_vals and m_energy else None
            # Kareus vs Nanobatch reduction: relative to Nanobatch baseline
            k_vs_n_time_red = (
                round(100.0 * (n_vals[0] - k_vals[0]) / n_vals[0], 2)
                if k_vals and n_vals and n_vals[0]
                else None
            )
            k_vs_n_energy_red = (
                round(100.0 * (n_vals[1] - k_vals[1]) / n_vals[1], 2)
                if k_vals and n_vals and n_vals[1]
                else None
            )

            rows.append(
                {
                    "model": model_name,
                    "config": config_key,
                    "megatron_time_s": round(m_time, 6),
                    "megatron_energy_J": round(m_energy, 6),
                    "kareus_time_s": k_time,
                    "kareus_energy_J": k_energy,
                    "time_reduction_pct": k_time_red,
                    "energy_reduction_pct": k_energy_red,
                    "nanobatch_time_s": n_time,
                    "nanobatch_energy_J": n_energy,
                    "kareus_vs_nanobatch_time_reduction_pct": k_vs_n_time_red,
                    "kareus_vs_nanobatch_energy_reduction_pct": k_vs_n_energy_red,
                }
            )

    if not rows:
        print("[!] No paired Megatron/Kareus runs found with sufficient data.")
        return

    df = pd.DataFrame(rows)
    df.sort_values(["model", "config"], inplace=True)
    # os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"[✓] Wrote {len(df)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()


