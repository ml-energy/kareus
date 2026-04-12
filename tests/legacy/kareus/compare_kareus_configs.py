"""Compare time and energy results across multiple Kareus configs.

This script reuses `parse_results` to:
  1. Automatically discover config subdirectories under a given root (e.g. `.../kareus/`).
  2. For each config, parse Zeus monitor files.
  3. Aggregate per-config metrics (max time across ranks, total energy across ranks).
  4. Write a small comparison CSV and print a human-readable summary.
"""

from __future__ import annotations

import argparse
import os
from glob import glob
from typing import Dict, Tuple

from parse_results import parse_results


def find_config_dirs(root: str) -> Dict[str, str]:
    """Find subdirectories under `root` that contain Zeus monitor files."""
    if not os.path.isdir(root):
        raise RuntimeError(f"Root directory does not exist: {root}")

    config_dirs: Dict[str, str] = {}

    for name in sorted(os.listdir(root)):
        full_path = os.path.join(root, name)
        if not os.path.isdir(full_path):
            continue

        # A valid config dir must contain at least one zeus monitor file
        pattern = os.path.join(full_path, "zeus_monitor_global_rank-*_local_rank-*.txt")
        if glob(pattern):
            config_dirs[name] = full_path

    if not config_dirs:
        raise RuntimeError(f"No config directories with Zeus monitor files found under {root}")

    return config_dirs


def compare_configs(
    root: str,
    num_ranks: int | None = None,
    warmup_iters: int = 10,
    profile_iters: int = 20,
    output_file: str = "kareus_config_comparison.csv",
) -> None:
    """Compare time and energy across all discovered configs.

    Args:
        root: Directory containing per-config subdirectories (e.g. `.../kareus`).
        num_ranks: Number of ranks to process. If None, auto-detect per config.
        warmup_iters: Number of warmup iterations to skip.
        profile_iters: Number of profiling iterations to use for averaging.
        output_file: Where to write the comparison CSV.
    """
    config_dirs = find_config_dirs(root)

    rows: Dict[str, Tuple[float, float]] = {}

    for config_name, config_path in config_dirs.items():
        print(f"Processing config '{config_name}' in {config_path}")

        results = parse_results(
            path=config_path,
            num_ranks=num_ranks,
            warmup_iters=warmup_iters,
            profile_iters=profile_iters,
        )

        if not results:
            print(f"  Warning: no usable results for config '{config_name}', skipping")
            continue

        max_time = max(x[0] for x in results.values()) if results else 0.0
        total_energy = sum(x[1] for x in results.values())
        rows[config_name] = (max_time, total_energy)

        print(f"  -> max_time={max_time:.6f}s, total_energy={total_energy:.6f}J")

    if not rows:
        print("No configs produced valid results; nothing to write.")
        return

    print(f"\nWriting comparison CSV to {output_file}")
    with open(output_file, "w") as f:
        f.write("config,max_time,total_energy\n")
        for cfg, (max_time, total_energy) in sorted(rows.items()):
            f.write(f"{cfg},{max_time},{total_energy}\n")

    print("\nSummary (sorted by total_energy):")
    for cfg, (max_time, total_energy) in sorted(rows.items(), key=lambda kv: kv[1][1]):
        print(f"  {cfg:>12s}: {max_time:.6f}s, {total_energy:.6f}J")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare time and energy across Kareus configs using Zeus monitor files"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=(
            "nemo_experiments/megatron_llama_3_2_3b/"
            "cp2_tp4_bs16_seq4096/kareus"
        ),
        help="Root directory containing config subdirectories (each with Zeus monitor files).",
    )
    parser.add_argument(
        "--num_ranks",
        type=int,
        default=None,
        help="Number of ranks to process. If not specified, auto-detect per config.",
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=10,
        help="Number of warmup iterations to skip.",
    )
    parser.add_argument(
        "--profile_iters",
        type=int,
        default=20,
        help="Number of profiling iterations to use for averaging.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="kareus_config_comparison.csv",
        help="Output CSV file path.",
    )

    args = parser.parse_args()

    compare_configs(
        root=args.root,
        num_ranks=args.num_ranks,
        warmup_iters=args.warmup_iters,
        profile_iters=args.profile_iters,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()


