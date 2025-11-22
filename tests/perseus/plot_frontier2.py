#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scan the cp2-tp4-bs16 frontier directory for the 3B LLaMA model, extract
average time and energy from Zeus monitor logs, and draw a scatter plot
(Y: energy, X: time).

This is a convenience wrapper around the generic frontier plotting logic
in `plot_frontier.py`, but with defaults tailored to:

  tests/perseus/nemo_experiments/megatron_llama_3_2_3b/cp2_tp4_bs16_seq4096/frontier

Assumptions:
- Each run directory under the target path contains Zeus monitor files
  matching: zeus_monitor_global_rank-<global>_local_rank-<local>.txt
- Energy returned by parse_results is per-GPU. We aggregate by mean
  across ranks to produce a single (time, energy) pair per run.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

# Ensure we can import sibling utility
if __name__ == "__main__":
    # Add this directory to sys.path so we can import parse_results.py
    import sys
    THIS_DIR = os.path.dirname(__file__)
    if THIS_DIR not in sys.path:
        sys.path.append(THIS_DIR)

from parse_results import parse_results  # noqa: E402


def _derive_title_suffix_from_out_path(out_path: Path) -> str:
    """
    Derive a short experiment descriptor for plot titles from the output path.

    Expected layout (by default):
      .../<model>/<config_dir>/frontier/energy_time*.png

    We take the parent of the 'frontier' directory (e.g. 'cp2_tp4_bs8_seq8192')
    and convert underscores to dashes so it appears as:
      'cp2-tp4-bs8-seq8192'
    """
    try:
        # e.g. out_path = .../cp2_tp4_bs8_seq8192/frontier/energy_time.png
        config_dir = out_path.parent.parent.name
        if not config_dir:
            return "3B"
        return f"3B, {config_dir.replace('_', '-')}"
    except Exception:
        # Fallback: keep a generic model-only label
        return "3B"


def collect_points(frontier_dir: Path) -> List[Tuple[str, float, float]]:
    """
    Walk the given frontier directory and collect (run_name, avg_time, avg_energy)
    for each child directory containing Zeus monitor logs.
    """
    points: List[Tuple[str, float, float]] = []

    if not frontier_dir.exists() or not frontier_dir.is_dir():
        raise FileNotFoundError(f"Frontier directory not found or not a directory: {frontier_dir}")

    for entry in sorted(frontier_dir.iterdir()):
        if not entry.is_dir():
            continue
        run_name = entry.name
        try:
            results: Dict[int, Tuple[float, float]] = parse_results(str(entry))
        except Exception as exc:
            print(f"[skip] {run_name}: failed to parse results: {exc}")
            continue

        if not results:
            print(f"[skip] {run_name}: empty results")
            continue

        times = np.array([v[0] for v in results.values()], dtype=float)
        energies = np.array([v[1] for v in results.values()], dtype=float)

        mean_time = float(np.mean(times))
        mean_energy = float(np.mean(energies))
        points.append((run_name, mean_time, mean_energy))

    return points


def filter_pareto(points: List[Tuple[str, float, float]]) -> List[Tuple[str, float, float]]:
    """
    Keep only points on the Pareto frontier (minimize time and energy).
    Input points are (label, time, energy). Returns the filtered subset.
    """
    n = len(points)
    if n == 0:
        return points
    times = np.array([p[1] for p in points], dtype=float)
    energies = np.array([p[2] for p in points], dtype=float)
    pareto_mask = np.ones(n, dtype=bool)
    for i in range(n):
        # A point i is dominated if exists j s.t. time_j <= time_i and energy_j <= energy_i
        # and at least one is strictly smaller.
        dominated = (
            (times <= times[i])
            & (energies <= energies[i])
            & ((times < times[i]) | (energies < energies[i]))
        )
        dominated[i] = False  # ignore self
        if np.any(dominated):
            pareto_mask[i] = False
    filtered = [pt for keep, pt in zip(pareto_mask.tolist(), points) if keep]
    return filtered


def plot_energy_time_multi(datasets: List[Tuple[str, List[Tuple[str, float, float]]]], out_path: Path) -> None:
    """
    Plot multiple datasets on the same axes.
    datasets: list of (label, points) where points are (run_name, time, energy)
    """
    plt.figure(figsize=(8, 6))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    markers = ["o", "s", "^", "x", "+"]

    any_points = False
    for di, (ds_label, points) in enumerate(datasets):
        if len(points) == 0:
            continue
        any_points = True
        labels = [p[0] for p in points]
        times = np.array([p[1] for p in points], dtype=float)
        energies = np.array([p[2] for p in points], dtype=float)

        c = colors[di % len(colors)]
        m = markers[di % len(markers)]
        plt.scatter(times, energies, c=c, s=35, marker=m, label=ds_label)

        # # Optional: annotate a couple of extremal points for each dataset
        # try:
        #     idx_min_time = int(np.argmin(times))
        #     idx_min_energy = int(np.argmin(energies))
        #     for idx in set([idx_min_time, idx_min_energy]):
        #         plt.annotate(labels[idx], (times[idx], energies[idx]),
        #                      textcoords="offset points", xytext=(6, 6), fontsize=8,
        #                      color="#444444")
        # except Exception:
        #     pass

    if not any_points:
        raise RuntimeError("No points to plot")

    plt.xlabel("Time (s)")
    plt.ylabel("Energy (J per GPU)")
    title_suffix = _derive_title_suffix_from_out_path(out_path)
    plt.title(f"Frontier: Energy vs Time ({title_suffix})")
    plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    plt.legend(loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved figure to: {out_path}")


def plot_effective_energy_time_multi(
    datasets: List[Tuple[str, List[Tuple[str, float, float]]]],
    p2p_power: float,
    out_path: Path,
) -> None:
    """
    Plot multiple datasets on the same axes with effective energy:
    effective_energy = energy - p2p_power * time
    datasets: list of (label, points) where points are (run_name, time, energy)
    """
    plt.figure(figsize=(8, 6))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    markers = ["o", "s", "^", "x", "+"]

    any_points = False
    for di, (ds_label, points) in enumerate(datasets):
        if len(points) == 0:
            continue
        any_points = True
        times = np.array([p[1] for p in points], dtype=float)
        energies = np.array([p[2] for p in points], dtype=float)
        eff_energies = energies - float(p2p_power) * times

        c = colors[di % len(colors)]
        m = markers[di % len(markers)]
        plt.scatter(times, eff_energies, c=c, s=35, marker=m, label=ds_label)

    if not any_points:
        raise RuntimeError("No points to plot")

    plt.xlabel("Time (s)")
    plt.ylabel("Effective Energy (J per GPU)")
    title_suffix = _derive_title_suffix_from_out_path(out_path)
    plt.title(f"Frontier: Effective Energy (E - P2P*Time) vs Time ({title_suffix})")
    plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    plt.legend(loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved figure to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frontier_dir",
        type=str,
        # default=str(
        #     Path(__file__).parent
        #     / "nemo_experiments/megatron_llama_3_2_3b/cp2_tp4_bs8_seq4096/frontier"
        # ),
        default="",
        help="Primary directory containing run subdirectories with Zeus monitor logs",
    )
    parser.add_argument(
        "--frontier_dir2",
        type=str,
        default=str(
            "/workspaces/Kareus/tests/kareus/" \
            "nemo_experiments/megatron_qwen3_1p7b/cp2_tp4_bs16_seq4096/kareus/frontier"
        ),
        # default="",
        help="Optional second directory to plot alongside the first",
    )
    parser.add_argument(
        "--frontier_dir3",
        type=str,
        # default=str(
        #     "/workspaces/Kareus/tests/kareus/" \
        #     "nemo_experiments/megatron_llama_3_2_3b/cp2_tp4_bs8_seq4096/nanobatch_perseus/frontier"
        # ),
        default="",
        help="Optional third directory to plot alongside the first two",
    )
    parser.add_argument(
        "--label1",
        type=str,
        default="Perseus",
        help="Legend label for the primary directory",
    )
    parser.add_argument(
        "--label2",
        type=str,
        default="Kareus",
        help="Legend label for the second directory",
    )
    parser.add_argument(
        "--label3",
        type=str,
        default="Nanobatch+Perseus",
        help="Legend label for the third directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        # default=str(
        #     Path(__file__).parent
        #     / "nemo_experiments/megatron_llama_3_2_3b/cp2_tp4_bs8_seq4096/frontier/energy_time.png"
        # ),
        default=str(
            "/workspaces/Kareus/tests/kareus/" \
            "nemo_experiments/megatron_qwen3_1p7b/cp2_tp4_bs16_seq4096/kareus/frontier"
        ),
        help="Path to save the Energy vs Time figure",
    )
    parser.add_argument(
        "--output_effective",
        type=str,
        # default=str(
        #     Path(__file__).parent
        #     / "nemo_experiments/megatron_llama_3_2_3b/cp2_tp4_bs8_seq4096/frontier/energy_time_effective.png"
        # ),
        default=str(
            "/workspaces/Kareus/tests/kareus/" \
            "nemo_experiments/megatron_qwen3_1p7b/cp2_tp4_bs16_seq4096/kareus/frontier"
        ),
        help="Path to save the Effective Energy vs Time figure",
    )
    parser.add_argument(
        "--p2p_power",
        type=float,
        default=85.0,
        help="GPU P2P blocking power in Watts (used for effective energy: E - p2p_power * T)",
    )
    args = parser.parse_args()

    frontier_dir = Path(args.frontier_dir)
    frontier_dir2 = Path(args.frontier_dir2) if args.frontier_dir2 else None
    frontier_dir3 = Path(args.frontier_dir3) if args.frontier_dir3 else None
    out_path = Path(args.output)
    out_path_eff = Path(args.output_effective)

    # Normalize output paths: if a directory is provided, append default filenames
    if out_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        out_path = out_path / "energy_time.png"
    if out_path_eff.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        out_path_eff = out_path_eff / "energy_time_effective.png"

    datasets: List[Tuple[str, List[Tuple[str, float, float]]]] = []
    pts1 = collect_points(frontier_dir)
    # pts1_sorted = sorted(filter_pareto(pts1), key=lambda t: t[1])
    pts1_sorted = sorted(pts1, key=lambda t: t[1])
    if len(pts1_sorted) > 0:
        datasets.append((args.label1, pts1_sorted))
        best = pts1_sorted[0]
        second = pts1_sorted[1] if len(pts1_sorted) > 1 else None
        msg = (
            f"[dir] {args.label1 or str(frontier_dir)} ({frontier_dir}) -> "
            f"fastest: {best[1]:.4f}s, energy: {best[2]:.2f}J (run: {best[0]})"
        )
        if second is not None:
            msg += (
                f", second: {second[1]:.4f}s, energy: {second[2]:.2f}J (run: {second[0]})"
            )
        print(msg)
    else:
        print(f"No valid runs found under {frontier_dir}")

    if frontier_dir2 is not None and frontier_dir2.exists():
        pts2 = collect_points(frontier_dir2)
        # pts2_sorted = sorted(filter_pareto(pts2), key=lambda t: t[1])
        pts2_sorted = sorted(pts2, key=lambda t: t[1])
        if len(pts2_sorted) > 0:
            # Append a single (label, points) tuple
            datasets.append((args.label2 or str(frontier_dir2), pts2_sorted))
            best = pts2_sorted[0]
            second = pts2_sorted[1] if len(pts2_sorted) > 1 else None
            msg = (
                f"[dir] {args.label2 or str(frontier_dir2)} ({frontier_dir2}) -> "
                f"fastest: {best[1]:.4f}s, energy: {best[2]:.2f}J (run: {best[0]})"
            )
            if second is not None:
                msg += (
                    f", second: {second[1]:.4f}s, energy: {second[2]:.2f}J (run: {second[0]})"
                )
            print(msg)
        else:
            print(f"No valid runs found under {frontier_dir2}")

    if frontier_dir3 is not None and frontier_dir3.exists():
        pts3 = collect_points(frontier_dir3)
        # pts3_sorted = sorted(filter_pareto(pts3), key=lambda t: t[1])
        pts3_sorted = sorted(pts3, key=lambda t: t[1])
        if len(pts3_sorted) > 0:
            # Append a single (label, points) tuple
            datasets.append((args.label3 or str(frontier_dir3), pts3_sorted))
            best = pts3_sorted[0]
            second = pts3_sorted[1] if len(pts3_sorted) > 1 else None
            msg = (
                f"[dir] {args.label3 or str(frontier_dir3)} ({frontier_dir3}) -> "
                f"fastest: {best[1]:.4f}s, energy: {best[2]:.2f}J (run: {best[0]})"
            )
            if second is not None:
                msg += (
                    f", second: {second[1]:.4f}s, energy: {second[2]:.2f}J (run: {second[0]})"
                )
            print(msg)
        else:
            print(f"No valid runs found under {frontier_dir3}")

    if len(datasets) == 0:
        print("No datasets to plot")
        return

    plot_energy_time_multi(datasets, out_path)
    # # Also plot effective energy with provided p2p power
    # try:
    #     plot_effective_energy_time_multi(datasets, float(args.p2p_power), out_path_eff)
    # except Exception as e:
    #     print(f"[warn] Failed to plot effective energy figure: {e}")


if __name__ == "__main__":
    main()


