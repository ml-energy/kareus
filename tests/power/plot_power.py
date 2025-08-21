#!/usr/bin/env python3
"""
Parse NeMo/Megatron power monitor CSV files and plot GPU power over time.

Input files pattern: power_monitor* (CSV with header: timestamp,power)
Each file corresponds to a single GPU (e.g., by local_rank in filename).

Usage:
  python plot_power.py --dir /path/to/experiment/dir --out power_over_time.pdf
"""

import argparse
import csv
import glob
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GPU power over time from power_monitor* CSV files")
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing power_monitor* files",
    )
    parser.add_argument(
        "--pattern",
        default="power_monitor*",
        help="Filename pattern to match (default: power_monitor*)",
    )
    parser.add_argument(
        "--instant_pattern",
        default="power_monitor_instant*",
        help="Pattern for instant power files (default: same as --pattern)",
    )
    parser.add_argument(
        "--average_pattern",
        default="power_monitor_average*",
        help="Pattern for average power files (default: avg_power_monitor*)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output image path (default: <dir>/power_over_time.pdf)",
    )
    parser.add_argument(
        "--zeus_pattern",
        default="zeus_monitor*",
        help="Zeus monitor filename pattern to detect training window (default: zeus_monitor*)",
    )
    parser.add_argument(
        "--start_step",
        type=int,
        default=3,
        help="1-based start step index for a contiguous window (inclusive). Use with --end_step",
    )
    parser.add_argument(
        "--end_step",
        type=int,
        default=5,
        help="1-based end step index for a contiguous window (inclusive). Use with --start_step",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Figure DPI (default: 160)",
    )
    parser.add_argument(
        "--resolution_s",
        type=float,
        default=0.1,
        help="Resolution in seconds for power series (default: 0.1)",
    )
    return parser.parse_args()


def extract_label_from_filename(filename: str) -> str:
    """Extract a concise GPU label from filename in the form:
    GPU [global G] [local L]

    Examples:
      power_monitor_global_rank-7_local_rank-3.txt -> GPU [global 7] [local 3]
    """
    base = os.path.basename(filename)
    local_match = re.search(r"local_rank-(\d+)", base)
    global_match = re.search(r"global_rank-(\d+)", base)
    parts = ["GPU"]
    if global_match:
        parts.append(f"[global {global_match.group(1)}]")
    if local_match:
        parts.append(f"[local {local_match.group(1)}]")
    if len(parts) > 1:
        return " ".join(parts)
    return base


def read_power_file(path: str) -> List[Tuple[float, float]]:
    """Read a CSV file with header 'timestamp,power' and return list of (timestamp, power)."""
    records: List[Tuple[float, float]] = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header_skipped = False
        for row in reader:
            if not header_skipped:
                header_skipped = True
                # tolerate files that already contain just data
                try:
                    if row and row[0] == "timestamp":
                        continue
                    # fallthrough: row may be data already
                except Exception:
                    pass
            if not row or len(row) < 2:
                continue
            try:
                ts = float(row[0])
                pw = float(row[1])
            except ValueError:
                # skip malformed rows
                continue
            records.append((ts, pw))
    return records


def load_all(dir_path: str, pattern: str) -> Dict[str, List[Tuple[float, float]]]:
    file_glob = os.path.join(dir_path, pattern)
    matched_files = sorted(glob.glob(file_glob))
    if not matched_files:
        raise FileNotFoundError(f"No files matched pattern: {file_glob}")

    label_to_records: Dict[str, List[Tuple[float, float]]] = {}
    for path in matched_files:
        label = extract_label_from_filename(path)
        records = read_power_file(path)
        if not records:
            continue
        label_to_records[label] = records

    if not label_to_records:
        raise RuntimeError("No valid power records parsed from matched files")

    return label_to_records


def align_to_zero(label_to_records: Dict[str, List[Tuple[float, float]]]) -> Dict[str, Tuple[List[float], List[float]]]:
    """Align timestamps so that the earliest timestamp across all files maps to t=0."""
    min_ts = min(min(ts for ts, _ in recs) for recs in label_to_records.values())
    aligned: Dict[str, Tuple[List[float], List[float]]] = {}
    for label, recs in label_to_records.items():
        times = [ts - min_ts for ts, _ in recs]
        powers = [pw for _, pw in recs]
        aligned[label] = (times, powers)
    return aligned


def configure_large_fonts(plt, base_size: int = 16) -> None:
    """Configure matplotlib to use larger, readable fonts across the board."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.sans-serif": ["DejaVu Sans"],
            # Ensure embedding of TrueType font in PDFs rather than Type 3 where possible
            "pdf.fonttype": 42,
            "font.size": base_size,
            "axes.titlesize": base_size + 2,
            "axes.labelsize": base_size,
            "xtick.labelsize": base_size - 2,
            "ytick.labelsize": base_size - 2,
            "legend.fontsize": base_size - 2,
            "figure.titlesize": base_size + 4,
        }
    )


def plot_power(
    aligned: Dict[str, Tuple[List[float], List[float]]],
    out_path: str,
    title: str,
    dpi: int = 160,
    resolution_s: float = 0.1,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise SystemExit(
            "matplotlib is required to plot the figure. Install with: pip install matplotlib"
        ) from exc

    configure_large_fonts(plt)
    plt.figure(figsize=(10, 6), dpi=dpi)
    for label, (times, powers) in sorted(aligned.items(), key=lambda kv: kv[0]):
        if not times:
            continue
        gr = extract_global_rank_from_label(label)
        display_label = f"GPU {gr}" if gr is not None else label
        plt.plot(times, powers, label=display_label, linewidth=1.2)

    plt.xlabel("Time (s from start)")
    plt.ylabel("Power (W)")
    # plt.title(title)
    plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def detect_training_window(dir_path: str, zeus_pattern: str) -> Optional[Tuple[float, float]]:
    """Scan zeus_monitor* files to find the first training_step start and the last training_step end.

    Returns (start_ts, end_ts) in absolute timestamps, or None if no training_step entries found.
    """
    file_glob = os.path.join(dir_path, zeus_pattern)
    files = sorted(glob.glob(file_glob))
    if not files:
        return None

    first_start: Optional[float] = None
    last_end: Optional[float] = None

    for path in files:
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            # skip header if present
            header_read = False
            for row in reader:
                if not header_read:
                    header_read = True
                    if row and row[0] == "start_time":
                        continue
                if not row or len(row) < 3:
                    continue
                try:
                    start_ts = float(row[0])
                    window_name = row[1]
                    elapsed = float(row[2])
                except (ValueError, IndexError):
                    continue
                if window_name == "training_step":
                    end_ts = start_ts + elapsed
                    if first_start is None or start_ts < first_start:
                        first_start = start_ts
                    if last_end is None or end_ts > last_end:
                        last_end = end_ts

    if first_start is None or last_end is None:
        return None
    return (first_start, last_end)


def collect_training_steps(dir_path: str, zeus_pattern: str) -> Dict[int, Tuple[float, float]]:
    """Collect global training step windows by ordinal across all zeus monitor files.

    For each file, we sort training_step entries by their start times and assign 1-based ordinals.
    For each ordinal i, we compute a global window as:
      start = min(start_i across files)
      end   = max(start_i + elapsed_i across files)

    Returns a mapping from step index -> (start_ts, end_ts).
    """
    file_glob = os.path.join(dir_path, zeus_pattern)
    files = sorted(glob.glob(file_glob))
    steps: Dict[int, Tuple[float, float]] = {}
    if not files:
        return steps

    # Accumulate per-file ordered lists
    for path in files:
        per_file_steps: List[Tuple[float, float]] = []
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            header_read = False
            for row in reader:
                if not header_read:
                    header_read = True
                    if row and row[0] == "start_time":
                        continue
                if not row or len(row) < 3:
                    continue
                try:
                    start_ts = float(row[0])
                    name = row[1]
                    elapsed = float(row[2])
                except (ValueError, IndexError):
                    continue
                if name == "training_step":
                    per_file_steps.append((start_ts, start_ts + elapsed))
        if not per_file_steps:
            continue
        per_file_steps.sort(key=lambda x: x[0])
        for idx, (s, e) in enumerate(per_file_steps, start=1):
            if idx in steps:
                cur_s, cur_e = steps[idx]
                steps[idx] = (min(cur_s, s), max(cur_e, e))
            else:
                steps[idx] = (s, e)
    return steps


def crop_and_align(
    label_to_records: Dict[str, List[Tuple[float, float]]],
    start_ts: float,
    end_ts: float,
) -> Dict[str, Tuple[List[float], List[float]]]:
    """Crop records to [start_ts, end_ts] and align timestamps so start_ts maps to t=0."""
    cropped: Dict[str, Tuple[List[float], List[float]]] = {}
    for label, recs in label_to_records.items():
        times: List[float] = []
        powers: List[float] = []
        for ts, pw in recs:
            if ts < start_ts or ts > end_ts:
                continue
            times.append(ts - start_ts)
            powers.append(pw)
        cropped[label] = (times, powers)
    return cropped


def compute_total_series(
    aligned: Dict[str, Tuple[List[float], List[float]]],
    resolution_s: float = 0.01,
) -> Tuple[List[float], List[float]]:
    """Quantize times to resolution (default 0.01s) and combine power across GPUs per bucket.

    For each GPU series, if multiple samples fall into the same bucket, they are averaged first.
    Then, bucketed averages are summed across GPUs to yield the total power per bucket.

    We avoid floating rounding issues by operating in integer ticks of 1/resolution.
    """
    if resolution_s <= 0:
        raise ValueError("resolution_s must be positive")
    tick = int(round(1.0 / resolution_s))

    # First pass: per-GPU bucket averages
    per_gpu_bucket_avgs: List[Dict[int, float]] = []
    for _, (times, powers) in aligned.items():
        sums: Dict[int, float] = defaultdict(float)
        counts: Dict[int, int] = defaultdict(int)
        for t, p in zip(times, powers):
            idx = int(round(t * tick))
            sums[idx] += p
            counts[idx] += 1
        if sums:
            avgs = {idx: (sums[idx] / counts[idx]) for idx in sums.keys()}
            per_gpu_bucket_avgs.append(avgs)

    # Second pass: sum averages across GPUs per bucket
    total_bucket_sum: Dict[int, float] = defaultdict(float)
    for avgs in per_gpu_bucket_avgs:
        for idx, val in avgs.items():
            total_bucket_sum[idx] += val

    if not total_bucket_sum:
        return [], []
    indices = sorted(total_bucket_sum.keys())
    out_times = [i / tick for i in indices]
    out_powers = [total_bucket_sum[i] for i in indices]
    return out_times, out_powers


def extract_local_rank_from_label(label: str) -> Optional[int]:
    m = re.search(r"\[local\s+(\d+)\]", label)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def extract_global_rank_from_label(label: str) -> Optional[int]:
    m = re.search(r"\[global\s+(\d+)\]", label)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def filter_aligned_by_local_ranks(
    aligned: Dict[str, Tuple[List[float], List[float]]],
    include_local_ranks: set,
) -> Dict[str, Tuple[List[float], List[float]]]:
    subset: Dict[str, Tuple[List[float], List[float]]] = {}
    for label, series in aligned.items():
        lr = extract_local_rank_from_label(label)
        if lr is not None and lr in include_local_ranks:
            subset[label] = series
    return subset


def filter_aligned_by_global_ranks(
    aligned: Dict[str, Tuple[List[float], List[float]]],
    include_global_ranks: set,
) -> Dict[str, Tuple[List[float], List[float]]]:
    subset: Dict[str, Tuple[List[float], List[float]]] = {}
    for label, series in aligned.items():
        gr = extract_global_rank_from_label(label)
        if gr is not None and gr in include_global_ranks:
            subset[label] = series
    return subset


def plot_group_totals(
    aligned: Dict[str, Tuple[List[float], List[float]]],
    out_path_base: str,
    title_suffix: str,
    dpi: int = 160,
    resolution_s: float = 0.1,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required to plot the figure. Install with: pip install matplotlib"
        ) from exc

    configure_large_fonts(plt)
    # Derive output path with _groups suffix
    root, ext = os.path.splitext(out_path_base)
    out_path = f"{root}_groups{ext or '.pdf'}"

    pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    plt.figure(figsize=(10, 6), dpi=dpi)
    for (i, j), color in zip(pairs, colors):
        aligned_pair = filter_aligned_by_global_ranks(aligned, {i, j})
        times, powers = compute_total_series(aligned_pair, resolution_s=resolution_s)
        if times:
            plt.plot(times, powers, label=f"Group {i}+{j}", linewidth=2.0, color=color)

    plt.xlabel("Time (s from start)")
    plt.ylabel("Power (W)")
    # plt.title(f"GPU Group Total Power\n{title_suffix}")
    plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    plt.legend(loc="best")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_node_totals(
    aligned: Dict[str, Tuple[List[float], List[float]]],
    out_path_base: str,
    title_suffix: str,
    dpi: int = 160,
    resolution_s: float = 0.1,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required to plot the figure. Install with: pip install matplotlib"
        ) from exc

    configure_large_fonts(plt)
    root, ext = os.path.splitext(out_path_base)
    out_path = f"{root}_nodes{ext or '.pdf'}"

    node0_ranks = {0, 1, 2, 3}
    node1_ranks = {4, 5, 6, 7}

    aligned_node0 = filter_aligned_by_global_ranks(aligned, node0_ranks)
    aligned_node1 = filter_aligned_by_global_ranks(aligned, node1_ranks)

    times0, power0 = compute_total_series(aligned_node0, resolution_s=resolution_s)
    times1, power1 = compute_total_series(aligned_node1, resolution_s=resolution_s)

    plt.figure(figsize=(10, 6), dpi=dpi)
    if times0:
        plt.plot(times0, power0, label="Node 0 (GPU 0-3)", linewidth=2.0, color="#1f77b4")
    if times1:
        plt.plot(times1, power1, label="Node 1 (GPU 4-7)", linewidth=2.0, color="#ff7f0e")

    plt.xlabel("Time (s from start)")
    plt.ylabel("Power (W)")
    # plt.title(f"GPU Node Total Power (0-3 vs 4-7)\n{title_suffix}")
    plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    plt.legend(loc="best")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_total_only(
    aligned: Dict[str, Tuple[List[float], List[float]]],
    out_path_base: str,
    title: str,
    dpi: int = 160,
    resolution_s: float = 0.1,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required to plot the figure. Install with: pip install matplotlib"
        ) from exc

    configure_large_fonts(plt)
    root, ext = os.path.splitext(out_path_base)
    out_path = f"{root}_total{ext or '.pdf'}"

    times, powers = compute_total_series(aligned, resolution_s=resolution_s)
    present = get_present_global_ranks(aligned)
    if present:
        print(f"[info] Total series computed over global ranks: {present} (count={len(present)})")
        if len(present) != 8:
            print("[warn] Expected 8 global ranks; total may be lower if some ranks are missing in the data.")
    plt.figure(figsize=(10, 6), dpi=dpi)
    if times:
        plt.plot(times, powers, label="Total power", linewidth=2.0, color="darkorange")
    plt.xlabel("Time (s from start)")
    plt.ylabel("Power (W)")
    # plt.title(title)
    plt.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    plt.legend(loc="best")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def load_all_safe(dir_path: str, pattern: Optional[str]) -> Dict[str, List[Tuple[float, float]]]:
    if not pattern:
        return {}
    try:
        return load_all(dir_path, pattern)
    except FileNotFoundError:
        return {}


def get_present_global_ranks(aligned: Dict[str, Tuple[List[float], List[float]]]) -> List[int]:
    ranks: List[int] = []
    for label in aligned.keys():
        gr = extract_global_rank_from_label(label)
        if gr is not None:
            ranks.append(gr)
    return sorted(set(ranks))


def debug_log_series(aligned: Dict[str, Tuple[List[float], List[float]]], tag: str) -> None:
    present = get_present_global_ranks(aligned)
    print(f"[info] {tag}: present global ranks = {present} (count={len(present)})")
    for label, (times, _) in sorted(aligned.items(), key=lambda kv: kv[0]):
        if times:
            print(f"  - {label}: samples={len(times)}, t_range=[{times[0]:.3f}, {times[-1]:.3f}]")
        else:
            print(f"  - {label}: samples=0")


def main() -> None:
    args = parse_args()
    dir_path = os.path.abspath(args.dir)
    out_path = args.out or os.path.join(dir_path, "power_over_time.pdf")

    # Handle instant and average patterns separately
    instant_pattern = args.instant_pattern or args.pattern
    avg_pattern = args.average_pattern
    label_to_records_instant = load_all_safe(dir_path, instant_pattern)
    label_to_records_avg = load_all_safe(dir_path, avg_pattern)

    steps_map = collect_training_steps(dir_path, args.zeus_pattern)
    if args.start_step is not None or args.end_step is not None:
        if args.start_step is None or args.end_step is None:
            raise SystemExit("Both --start_step and --end_step must be provided together.")
        if args.start_step <= 0 or args.end_step <= 0:
            raise SystemExit("--start_step and --end_step must be positive (1-based).")
        if args.start_step > args.end_step:
            raise SystemExit("--start_step must be <= --end_step.")
        needed = list(range(args.start_step, args.end_step + 1))
        missing = [i for i in needed if i not in steps_map]
        if missing:
            available = ", ".join(str(i) for i in sorted(steps_map.keys())) or "none"
            raise SystemExit(f"Requested step window not found (missing steps: {missing}). Available steps: {available}")
        start_ts = min(steps_map[i][0] for i in needed)
        end_ts = max(steps_map[i][1] for i in needed)
        # Instant
        if label_to_records_instant:
            cropped_inst = crop_and_align(label_to_records_instant, start_ts, end_ts)
            debug_log_series(cropped_inst, tag="instant after crop")
            title_inst = (
                f"Instant GPU Power (steps {args.start_step}-{args.end_step})\n"
                f"{os.path.basename(dir_path)} | elapsed={end_ts - start_ts:.3f}s"
            )
            out_inst = out_path.replace(".pdf", "_instant.pdf")
            plot_power(cropped_inst, out_inst, title_inst, dpi=args.dpi)
            plot_total_only(cropped_inst, out_inst, title_inst, dpi=args.dpi)
            plot_group_totals(cropped_inst, out_inst, title_suffix=os.path.basename(dir_path), dpi=args.dpi)
            plot_node_totals(cropped_inst, out_inst, title_suffix=os.path.basename(dir_path), dpi=args.dpi)
        # Average
        if label_to_records_avg:
            cropped_avg = crop_and_align(label_to_records_avg, start_ts, end_ts)
            debug_log_series(cropped_avg, tag="average after crop")
            title_avg = (
                f"Average GPU Power (steps {args.start_step}-{args.end_step})\n"
                f"{os.path.basename(dir_path)} | elapsed={end_ts - start_ts:.3f}s"
            )
            out_avg = out_path.replace(".pdf", "_average.pdf")
            plot_power(cropped_avg, out_avg, title_avg, dpi=args.dpi)
            plot_total_only(cropped_avg, out_avg, title_avg, dpi=args.dpi)
            plot_group_totals(cropped_avg, out_avg, title_suffix=os.path.basename(dir_path), dpi=args.dpi)
            plot_node_totals(cropped_avg, out_avg, title_suffix=os.path.basename(dir_path), dpi=args.dpi)
    else:
        if steps_map:
            # Use full training window across all steps
            start_ts = min(s for s, _ in steps_map.values())
            end_ts = max(e for _, e in steps_map.values())
            if label_to_records_instant:
                cropped_inst = crop_and_align(label_to_records_instant, start_ts, end_ts)
                debug_log_series(cropped_inst, tag="instant after crop")
                title_inst = (
                    f"Instant GPU Power (training window)\n"
                    f"{os.path.basename(dir_path)} | elapsed={end_ts - start_ts:.3f}s"
                )
                out_inst = out_path.replace(".pdf", "_instant.pdf")
                plot_power(cropped_inst, out_inst, title_inst, dpi=args.dpi, resolution_s=args.resolution_s)
                plot_total_only(cropped_inst, out_inst, title_inst, dpi=args.dpi, resolution_s=args.resolution_s)
                plot_group_totals(cropped_inst, out_inst, title_suffix=os.path.basename(dir_path), dpi=args.dpi, resolution_s=args.resolution_s)
                plot_node_totals(cropped_inst, out_inst, title_suffix=os.path.basename(dir_path), dpi=args.dpi, resolution_s=args.resolution_s)
            if label_to_records_avg:
                cropped_avg = crop_and_align(label_to_records_avg, start_ts, end_ts)
                debug_log_series(cropped_avg, tag="average after crop")
                title_avg = (
                    f"Average GPU Power (training window)\n"
                    f"{os.path.basename(dir_path)} | elapsed={end_ts - start_ts:.3f}s"
                )
                out_avg = out_path.replace(".pdf", "_average.pdf")
                plot_power(cropped_avg, out_avg, title_avg, dpi=args.dpi, resolution_s=args.resolution_s)
                plot_total_only(cropped_avg, out_avg, title_avg, dpi=args.dpi, resolution_s=args.resolution_s)
                plot_group_totals(cropped_avg, out_avg, title_suffix=os.path.basename(dir_path), dpi=args.dpi, resolution_s=args.resolution_s)
                plot_node_totals(cropped_avg, out_avg, title_suffix=os.path.basename(dir_path), dpi=args.dpi, resolution_s=args.resolution_s)
        else:
            if label_to_records_instant:
                aligned_inst = align_to_zero(label_to_records_instant)
                debug_log_series(aligned_inst, tag="instant aligned full")
                title_inst = f"Instant GPU Power\n{os.path.basename(dir_path)}"
                out_inst = out_path.replace(".pdf", "_instant.pdf")
                plot_power(aligned_inst, out_inst, title_inst, dpi=args.dpi, resolution_s=args.resolution_s)
                plot_total_only(aligned_inst, out_inst, title_inst, dpi=args.dpi, resolution_s=args.resolution_s)
                plot_group_totals(aligned_inst, out_inst, title_suffix=os.path.basename(dir_path), dpi=args.dpi, resolution_s=args.resolution_s)
                plot_node_totals(aligned_inst, out_inst, title_suffix=os.path.basename(dir_path), dpi=args.dpi, resolution_s=args.resolution_s)
            if label_to_records_avg:
                aligned_avg = align_to_zero(label_to_records_avg)
                debug_log_series(aligned_avg, tag="average aligned full")
                title_avg = f"Average GPU Power\n{os.path.basename(dir_path)}"
                out_avg = out_path.replace(".pdf", "_average.pdf")
                plot_power(aligned_avg, out_avg, title_avg, dpi=args.dpi, resolution_s=args.resolution_s)
                plot_total_only(aligned_avg, out_avg, title_avg, dpi=args.dpi, resolution_s=args.resolution_s)
                plot_group_totals(aligned_avg, out_avg, title_suffix=os.path.basename(dir_path), dpi=args.dpi, resolution_s=args.resolution_s)
                plot_node_totals(aligned_avg, out_avg, title_suffix=os.path.basename(dir_path), dpi=args.dpi, resolution_s=args.resolution_s)
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()


