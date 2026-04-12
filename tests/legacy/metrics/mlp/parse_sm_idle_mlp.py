import os
import re
import glob
import argparse
from collections import defaultdict
from typing import Dict, Tuple, List, Any, Optional, Set

import pandas as pd
import matplotlib.pyplot as plt


ConfigKey = Tuple[int, int, int, int]


def parse_config_from_filename(filename: str) -> Optional[ConfigKey]:
    basename = os.path.basename(filename)
    # profile_{overlap_start}_{overlap_end}_{sm_num}_{block_size}.csv
    m = re.match(r"profile_(-?\d+)_(-?\d+)_(\d+)_(\d+)\.csv$", basename)
    if not m:
        return None
    overlap_start, overlap_end, sm_num, block_size = map(int, m.groups())
    return overlap_start, overlap_end, sm_num, block_size


def compute_sm_idle_ratio(trace_df: pd.DataFrame) -> Optional[float]:
    sm_col = "SMs Active [Throughput %]"
    if sm_col not in trace_df.columns:
        print(f"SMs Active [Throughput %] not found in {trace_df.columns}")
        return None

    values = pd.to_numeric(trace_df[sm_col], errors="coerce").dropna()
    if values.empty:
        return None

    # Values are percentages in [0, 100]. Idle when SM Active < 30%.
    idle_ratio = float((values < 25.0).sum()) / float(len(values))
    return idle_ratio


def read_metrics_means(stats_csv: str) -> Dict[str, float]:
    if not os.path.exists(stats_csv):
        return {}
    df = pd.read_csv(stats_csv)
    if not {"Metric", "Mean"}.issubset(df.columns):
        return {}
    means: Dict[str, float] = {}
    for _, row in df.iterrows():
        metric = str(row["Metric"]).strip()
        means[metric] = float(row["Mean"]) if pd.notna(row["Mean"]) else None
    return means


def compute_sm_compute_idle_ratio(trace_df: pd.DataFrame) -> Optional[float]:
    # Ensure SM Instructions [%] exists or derive it
    inst_col = "SM Instructions [Throughput %]"
    if inst_col not in trace_df.columns:
        issue_col = "SM Issue [Throughput %]"
        tensor_col = "Tensor Active [Throughput %]"
        if issue_col in trace_df.columns and tensor_col in trace_df.columns:
            issue_vals = pd.to_numeric(trace_df[issue_col], errors="coerce")
            tensor_vals = pd.to_numeric(trace_df[tensor_col], errors="coerce")
            inst_vals = issue_vals + tensor_vals
        else:
            return None
    else:
        inst_vals = pd.to_numeric(trace_df[inst_col], errors="coerce")

    inst_vals = inst_vals.dropna()
    if inst_vals.empty:
        return None

    # Idle when SM Instructions < 10%
    return float((inst_vals < 10.0).sum()) / float(len(inst_vals))


def collect_profile_data(results_dir: str) -> Tuple[Dict[ConfigKey, Dict[str, float]], Dict[ConfigKey, float], Dict[ConfigKey, float], Set[str]]:
    """Collect per-config average metrics and SM idle ratio from profile CSVs.

    Returns:
        metrics_means_map: config -> {metric_name: mean}
        sm_idle_ratio_map: config -> SM active < 30% ratio
        sm_compute_idle_ratio_map: config -> SM instructions < 10% ratio
        all_metric_names: set of all metric names observed
    """
    metrics_means_map: Dict[ConfigKey, Dict[str, float]] = {}
    sm_idle_ratio_map: Dict[ConfigKey, float] = {}
    sm_compute_idle_ratio_map: Dict[ConfigKey, float] = {}
    all_metric_names: Set[str] = set()

    pattern = os.path.join(results_dir, "profile_*.csv")
    for csv_file in sorted(glob.glob(pattern)):
        if csv_file.endswith(".metrics_stats.csv"):
            continue

        config = parse_config_from_filename(csv_file)
        if config is None:
            continue

        # Trace for idle ratio
        # try:
        trace_df = pd.read_csv(csv_file)
        # except Exception:
        #     trace_df = pd.DataFrame()

        idle_ratio = compute_sm_idle_ratio(trace_df)
        if idle_ratio is None:
            print(f"idle_ratio is None for {config}")
            continue
        sm_idle_ratio_map[config] = idle_ratio

        compute_idle_ratio = compute_sm_compute_idle_ratio(trace_df)
        sm_compute_idle_ratio_map[config] = compute_idle_ratio

        # Metrics means from metrics_stats if available; fallback to means from trace
        stats_csv = csv_file[:-4] + ".metrics_stats.csv"
        means = read_metrics_means(stats_csv)
        # if not means and not trace_df.empty:
        #     # Fallback: compute means across numeric columns (excluding timestamp)
        #     numeric_df = trace_df.copy()
        #     if "timestamp" in numeric_df.columns:
        #         numeric_df = numeric_df.drop(columns=["timestamp"])
        #     numeric_df = numeric_df.apply(pd.to_numeric, errors="coerce")
        #     means = numeric_df.mean(numeric_only=True).to_dict()

        metrics_means_map[config] = means
        all_metric_names.update(means.keys())

    return metrics_means_map, sm_idle_ratio_map, sm_compute_idle_ratio_map, all_metric_names


def parse_energy_logs(log_dir: str) -> Dict[ConfigKey, Dict[str, float]]:
    """Parse energy logs under log_dir and extract 0:time and 0:total energy per config.

    Expects a single CSV at <log_dir>/<frequency>/energy_results.csv.

    Returns:
        config -> {"time_s": time_0, "total_energy_J": energy_0}
    """

    path = os.path.join(log_dir, "energy_results.csv")
    # matches = glob.glob(pattern, recursive=True)
    # if not matches:
    #     return {}

    # path = matches[0]
    # try:
    df = pd.read_csv(path)
    # except Exception:
    #     return {}

    time_col = "0:time (s)"
    energy_col = "0:total energy (J)"
    base_cols = ["overlap_start", "overlap_end", "comm_sm_number", "comm_block_size"]
    if not set(base_cols).issubset(df.columns) or time_col not in df.columns or energy_col not in df.columns:
        return {}

    result: Dict[ConfigKey, Dict[str, float]] = {}
    for _, row in df.iterrows():
        # try:
        config: ConfigKey = (
            int(row["overlap_start"]),
            int(row["overlap_end"]),
            int(row["comm_sm_number"]),
            int(row["comm_block_size"]),
        )
        # except Exception:
        #     continue

        time_val = float(row[time_col]) if pd.notna(row[time_col]) else None
        energy_val = float(row[energy_col]) if pd.notna(row[energy_col]) else None
        result[config] = {"time_s": time_val, "total_energy_J": energy_val}

    return result


def merge_and_write(
    output_csv: str,
    metrics_means_map: Dict[ConfigKey, Dict[str, float]],
    sm_idle_ratio_map: Dict[ConfigKey, float],
    energy_map: Dict[ConfigKey, Dict[str, float]],
    sm_compute_idle_ratio_map: Dict[ConfigKey, float],
    all_metric_names: Set[str],
) -> None:
    all_configs: Set[ConfigKey] = set(metrics_means_map.keys()) | set(energy_map.keys()) | set(sm_idle_ratio_map.keys()) | set(sm_compute_idle_ratio_map.keys())
    metric_names_sorted = sorted(all_metric_names)

    rows: List[Dict[str, Any]] = []
    for config in sorted(all_configs):
        overlap_start, overlap_end, sm_num, block_size = config
        row: Dict[str, Any] = {
            "overlap_start": overlap_start,
            "overlap_end": overlap_end,
            "sm_num": sm_num,
            "block_size": block_size,
        }
        # SM idle ratio
        row["sm_idle_ratio"] = sm_idle_ratio_map.get(config)
        row["sm_compute_idle_ratio"] = sm_compute_idle_ratio_map.get(config)
        # Time/Energy
        energy = energy_map.get(config, {})
        row["time_s"] = energy.get("time_s")
        row["total_energy_J"] = energy.get("total_energy_J")
        # Metrics means
        means = metrics_means_map.get(config, {})
        for metric in metric_names_sorted:
            row[f"avg:{metric}"] = means.get(metric)

        rows.append(row)

    if not rows:
        print("No data found to write.")
        return

    df = pd.DataFrame(rows)
    # Sort columns: config keys first, then sm_idle_ratio, time, energy, then metrics
    leading_cols = ["overlap_start", "overlap_end", "sm_num", "block_size", "sm_idle_ratio", "sm_compute_idle_ratio", "time_s", "total_energy_J"]
    metric_cols = [c for c in df.columns if c.startswith("avg:")]
    ordered_cols = [c for c in leading_cols if c in df.columns] + sorted(metric_cols)
    df = df[ordered_cols]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[✓] Summary written to {output_csv}")

    # Draw figures
    try:
        plots_dir = os.path.dirname(output_csv)
        # Plot 1: sm_idle_ratio vs total_energy_J
        xy = df[["sm_idle_ratio", "total_energy_J"]].dropna()
        if not xy.empty:
            plt.figure()
            plt.scatter(xy["sm_idle_ratio"], xy["total_energy_J"], s=20, alpha=0.7)
            plt.xlabel("SM idle ratio (<30% active)")
            plt.ylabel("Total energy (J)")
            plt.title("Energy vs SM idle ratio - MLP")
            plt.grid(True, alpha=0.3)
            out_path = os.path.join(plots_dir, "sm_idle_vs_energy_mlp.png")
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()
            print(f"[✓] Figure written to {out_path}")

        # Plot 2: sm_compute_idle_ratio vs total_energy_J
        xy = df[["sm_compute_idle_ratio", "total_energy_J"]].dropna()
        if not xy.empty:
            plt.figure()
            plt.scatter(xy["sm_compute_idle_ratio"], xy["total_energy_J"], s=20, alpha=0.7)
            plt.xlabel("SM compute idle ratio (instructions <10%)")
            plt.ylabel("Total energy (J)")
            plt.title("Energy vs SM compute idle ratio - MLP")
            plt.grid(True, alpha=0.3)
            out_path = os.path.join(plots_dir, "sm_compute_idle_vs_energy_mlp.png")
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()
            print(f"[✓] Figure written to {out_path}")
    except Exception as e:
        print(f"[!] Plotting skipped due to error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse MLP profiling and energy results")
    parser.add_argument("--frequency", "-f", type=str, default="default", help="Frequency tag used in results/logs")
    parser.add_argument("--world_size", "-w", type=int, default=2)
    parser.add_argument("--batch_size", "-b", type=int, default=8)
    parser.add_argument("--seq_len", "-s", type=int, default=4096)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional explicit output CSV path",
    )
    args = parser.parse_args()

    tp_bs_seq = f"tp{args.world_size}-bs{args.batch_size}-seq{args.seq_len}"
    profile_dir = os.path.join("results", tp_bs_seq, str(args.frequency))
    logs_dir = os.path.join("logs", tp_bs_seq, str(args.frequency))

    output_csv = (
        args.output
        if args.output
        else os.path.join(logs_dir, "sm_idle_ratio.csv")
    )

    metrics_means_map, sm_idle_ratio_map, sm_compute_idle_ratio_map, all_metric_names = collect_profile_data(profile_dir)
    energy_map = parse_energy_logs(logs_dir)
    merge_and_write(output_csv, metrics_means_map, sm_idle_ratio_map, energy_map, sm_compute_idle_ratio_map, all_metric_names)


if __name__ == "__main__":
    main()
