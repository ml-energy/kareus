import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import pandas as pd


def find_kernel_min_ns(kernel_csv_path: str, kernel_name: str = "allreduceKernelEntryPointBF16") -> Optional[float]:
    """
    Read an Nsight 'cuda_gpu_kern_sum' CSV and return the 'Min (ns)' for a given kernel name.

    Returns None if the kernel row is not present.
    """
    try:
        df = pd.read_csv(kernel_csv_path)
    except Exception:
        return None

    if "Name" not in df.columns:
        return None

    kernel_rows = df[df["Name"] == kernel_name]
    if kernel_rows.empty:
        return None

    # If multiple rows exist, take the minimum of their "Min (ns)" values for robustness
    min_ns_col = "Min (ns)"
    if min_ns_col not in kernel_rows.columns:
        return None

    try:
        min_ns_value = float(kernel_rows[min_ns_col].min())
    except Exception:
        return None

    return min_ns_value


def parse_profile_filename(filename: str) -> Optional[Tuple[int, int, int, int]]:
    """
    Parse overlap_start, overlap_end, sm_number, block_size from filenames like:
    profile_-1_-1_2_512.csv_cuda_gpu_kern_sum.csv

    Returns (overlap_start, overlap_end, sm_number, block_size) or None if no match.
    """
    # Capture the 4 integers before the first .csv in the name
    match = re.search(r"profile_(-?\d+)_(-?\d+)_(\d+)_(\d+)\.csv", filename)
    if not match:
        return None
    overlap_start = int(match.group(1))
    overlap_end = int(match.group(2))
    sm_number = int(match.group(3))
    block_size = int(match.group(4))
    return overlap_start, overlap_end, sm_number, block_size


def collect_comm_effect_factors(profile_dir: str) -> List[Dict[str, float]]:
    """
    Walk the profile_dir, find all '*cuda_gpu_kern_sum.csv' files, compute
    communication effect factor = sm_number * Min(ns of allreduceKernelEntryPointBF16).

    Returns a list of dicts with keys: overlap_start, overlap_end, sm_number, block_size, comm_effect_factor.
    """
    records: List[Dict[str, float]] = []
    for root, _dirs, files in os.walk(profile_dir):
        for fname in files:
            if not fname.endswith("cuda_gpu_kern_sum.csv"):
                continue
            parsed = parse_profile_filename(fname)
            if not parsed:
                print(f"Skipping {fname} because it does not match the expected format")
                continue
            overlap_start, overlap_end, sm_number, block_size = parsed
            csv_path = os.path.join(root, fname)
            min_ns = find_kernel_min_ns(csv_path)
            if min_ns is None:
                print(f"Skipping {fname} because it does not contain the kernel allreduceKernelEntryPointBF16")
                continue

            comm_effect_factor = float(sm_number) * float(min_ns)
            records.append({
                "overlap_start": overlap_start,
                "overlap_end": overlap_end,
                "sm_number": sm_number,
                "block_size": block_size,
                "comm_effect_factor": comm_effect_factor,
            })
    return records


def read_energy_times_energys(energy_csv_path: str) -> pd.DataFrame:
    """
    Read energy_results.csv and keep relevant columns for lookup.
    """
    df = pd.read_csv(energy_csv_path)
    required_cols = [
        "overlap_start",
        "overlap_end",
        "comm_sm_number",
        "comm_block_size",
        "0:time (s)",
        "0:total energy (J)",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"energy_results.csv missing columns: {missing}")
    return df[required_cols].copy()


def join_comm_effect_with_time_energy(records: List[Dict[str, float]], energy_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each comm effect record, find corresponding time(s) from energy_df and return a dataframe
    with columns: overlap_start, overlap_end, sm_number, block_size, comm_effect_factor, time_s
    """
    if not records:
        return pd.DataFrame(columns=[
            "overlap_start",
            "overlap_end",
            "sm_number",
            "block_size",
            "comm_effect_factor",
            "time_s",
        ])

    df_records = pd.DataFrame(records)
    # Prepare energy df for join by matching column names
    energy = energy_df.rename(columns={
        "comm_sm_number": "sm_number",
        "comm_block_size": "block_size",
        "0:time (s)": "time_s",
        "0:total energy (J)": "total_energy_J",
    })

    merged = pd.merge(
        df_records,
        energy,
        how="inner",
        on=["overlap_start", "overlap_end", "sm_number", "block_size"],
    )
    return merged[[
        "overlap_start",
        "overlap_end",
        "sm_number",
        "block_size",
        "comm_effect_factor",
        "time_s",
        "total_energy_J",
    ]]


def plot_comm_effect_vs_time(data: pd.DataFrame, output_path: str) -> None:
    """
    Create a scatter plot of communication effect factor (x) vs time (s) (y).
    """
    if data.empty:
        raise ValueError("No data to plot (empty dataframe)")

    plt.figure(figsize=(8, 5))
    plt.scatter(data["comm_effect_factor"], data["time_s"], s=22, alpha=0.8)
    plt.xlabel("Communication effect factor (SM number × Time of communication kernel)")
    plt.ylabel("Time (s)")
    plt.title("Communication effect factor vs Time - MLP")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_comm_effect_vs_energy(data: pd.DataFrame, output_path: str) -> None:
    """
    Create a scatter plot of communication effect factor (x) vs total energy (J) (y).
    Assumes 'total_energy_J' exists in data.
    """
    if data.empty:
        raise ValueError("No data to plot (empty dataframe)")

    if "total_energy_J" not in data.columns:
        raise ValueError("Column 'total_energy_J' not found in data for energy plot")

    plt.figure(figsize=(8, 5))
    plt.scatter(data["comm_effect_factor"], data["total_energy_J"], s=22, alpha=0.8, color="tab:orange")
    plt.xlabel("Communication effect factor (SM number × Time of communication kernel)")
    plt.ylabel("Total energy (J)")
    plt.title("Communication effect factor vs Energy - MLP")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def read_summary_sm_idle(summary_csv_path: str) -> pd.DataFrame:
    """
    Read summary_mlp_{freq}.csv and return minimal columns for join and filtering by sm_idle_ratio.
    """
    if not os.path.isfile(summary_csv_path):
        raise FileNotFoundError(f"summary csv not found: {summary_csv_path}")
    df = pd.read_csv(summary_csv_path)
    required_cols = [
        "overlap_start",
        "overlap_end",
        "sm_num",
        "block_size",
        "sm_idle_ratio",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"summary csv missing columns: {missing}")

    df = df[required_cols].copy()
    # Normalize types and names to match joins
    for col in ["overlap_start", "overlap_end", "sm_num", "block_size"]:
        df[col] = df[col].astype(int)
    df = df.rename(columns={"sm_num": "sm_number"})
    return df


def process_one_frequency(profile_dir: str, energy_csv: str, output_path: Optional[str] = None, save_csv_path: Optional[str] = None) -> Tuple[str, str]:
    """
    Process a single frequency directory: collect comm effect factors, join with energy times,
    save merged CSV and plot. Returns (save_csv_path, output_path).
    """
    if not os.path.isdir(profile_dir):
        raise FileNotFoundError(f"profile_dir not found: {profile_dir}")
    if not os.path.isfile(energy_csv):
        raise FileNotFoundError(f"energy_csv not found: {energy_csv}")

    records = collect_comm_effect_factors(profile_dir)
    energy_df = read_energy_times_energys(energy_csv)
    merged = join_comm_effect_with_time_energy(records, energy_df)

    if merged.empty:
        raise RuntimeError("No matching records found between profile CSVs and energy_results.csv")

    energy_dir = os.path.dirname(os.path.abspath(energy_csv))
    if output_path is None:
        output_path = os.path.join(energy_dir, "comm_effect_vs_time.png")
    if save_csv_path is None:
        save_csv_path = os.path.join(energy_dir, "comm_effect_vs_time.csv")

    # Read per-frequency summary and filter on sm_idle_ratio < 0.1 for plotting
    energy_dir = os.path.dirname(os.path.abspath(energy_csv))
    freq_name = os.path.basename(energy_dir)
    summary_csv = os.path.join(energy_dir, f"summary_mlp_{freq_name}.csv")
    summary_df = read_summary_sm_idle(summary_csv)

    merged_with_summary = pd.merge(
        merged,
        summary_df,
        how="inner",
        on=["overlap_start", "overlap_end", "sm_number", "block_size"],
    )

    if merged_with_summary.empty:
        raise RuntimeError("No matching records after merging with summary (check keys overlap)")

    filtered = merged_with_summary[merged_with_summary["sm_idle_ratio"] < 0.03].copy()
    if filtered.empty:
        raise RuntimeError("No records with sm_idle_ratio < 0.1 to plot")

    # Save data: full merged (with sm_idle_ratio) and filtered subset
    merged_with_summary.to_csv(save_csv_path, index=False)
    base_csv, ext_csv = os.path.splitext(save_csv_path)
    filtered_csv_path = f"{base_csv}.filtered{ext_csv or '.csv'}"
    filtered.to_csv(filtered_csv_path, index=False)

    # Plot only filtered points (time)
    plot_comm_effect_vs_time(filtered, output_path)

    # Plot only filtered points (energy)
    if output_path:
        base_png, ext_png = os.path.splitext(output_path)
        energy_output_path = f"{base_png}_energy{ext_png or '.png'}"
    else:
        energy_output_path = os.path.join(energy_dir, "comm_effect_vs_energy.png")
    plot_comm_effect_vs_energy(filtered, energy_output_path)

    return save_csv_path, output_path


def main():
    parser = argparse.ArgumentParser(description="Parse Nsight kernel CSVs to compute communication effect factor and plot vs time.")
    # New directory-structure-driven args
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size, e.g., 2")
    parser.add_argument("--bs", type=int, default=16, help="Batch size, e.g., 16")
    parser.add_argument("--seq", type=int, default=4096, help="Sequence length, e.g., 4096")
    parser.add_argument("--freq", type=str, default=1300, help="Frequency label, e.g., '1500' or 'default'")
    parser.add_argument(
        "--base_dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Base directory containing 'profile_result' and 'logs' (defaults to script directory).",
    )

    parser.add_argument("--output", default=None, help="Path to save the output plot (PNG). Defaults next to energy_csv.")
    parser.add_argument("--save_csv", default=None, help="Optional path to save the merged data as CSV.")
    args = parser.parse_args()

    tp_bs_seq = f"tp{args.tp}-bs{args.bs}-seq{args.seq}"
    profile_root = os.path.join(args.base_dir, "profile_result", tp_bs_seq)
    logs_root = os.path.join(args.base_dir, "logs", tp_bs_seq)

    profile_dir = os.path.join(profile_root, str(args.freq))
    energy_csv = os.path.join(logs_root, str(args.freq), "energy_results.csv")

    output_path = args.output
    save_csv_path = args.save_csv
    csv_path, png_path = process_one_frequency(profile_dir, energy_csv, output_path, save_csv_path)
    print(f"Saved merged data to: {csv_path}")
    print(f"Saved plot to: {png_path}")


if __name__ == "__main__":
    main()


