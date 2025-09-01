import os
import argparse
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt


def find_all_runs(results_dir):
    runs = []
    if not os.path.isdir(results_dir):
        return runs
    for name in os.listdir(results_dir):
        if not name.endswith(".sqlite"):
            continue
        base = name[:-7]  # strip .sqlite
        # Expect name: profile_<start>_<end>_<sm>_<bs>
        if not base.startswith("profile_"):
            continue
        parts = base.split("_")
        if len(parts) < 5:
            continue
        try:
            overlap_start = int(parts[1])
            overlap_end = int(parts[2])
            sm_num = int(parts[3])
            block_size = int(parts[4])
        except ValueError:
            continue
        csv_path = os.path.join(results_dir, f"{base}.csv_cuda_gpu_kern_sum.csv")
        if not os.path.exists(csv_path):
            # Fallback to plain .csv if suffixed file not found
            alt_csv = os.path.join(results_dir, f"{base}.csv")
            csv_path = alt_csv if os.path.exists(alt_csv) else None
        runs.append({
            "sqlite": os.path.join(results_dir, name),
            "csv": csv_path,
            "png": os.path.join(results_dir, f"{base}.timeline.png"),
            "overlap_start": overlap_start,
            "overlap_end": overlap_end,
            "sm_num": sm_num,
            "block_size": block_size,
            "basename": base,
        })
    return sorted(runs, key=lambda x: x["basename"])    


def draw_timeline_from_csv(csv_path, out_png, overlap_window, title_suffix=None, draw=True):
    if csv_path is None or not os.path.exists(csv_path):
        return False, None

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"Empty csv: {csv_path}")
        return False, None

    # Normalize column names
    cols = {c.lower(): c for c in df.columns}
    col_kernel = cols.get('name'.lower())
    col_device = cols.get('device'.lower()) or cols.get('device id'.lower())
    col_min = cols.get('min (ns)'.lower())
    col_avg = cols.get('avg (ns)'.lower())
    col_total = cols.get('total (ns)'.lower())
    if not (col_kernel and col_min and col_avg):
        print(f"Missing columns: {col_kernel}, {col_min}, {col_avg}")
        return False, None

    # Exclude FillFunctor
    df = df[~df[col_kernel].str.contains('FillFunctor', na=False)]

    # AllReduce latency: use Min (ns)
    ar_rows = df[df[col_kernel] == 'allreduceKernelEntryPointBF16']
    if ar_rows.empty:
        print(f"No AllReduce kernel found: {csv_path}")
        return False, None
    # # If multiple devices, choose device with smallest total time for AR; else global min
    # if col_device and col_total and ar_rows[col_total].notna().any():
    #     ar_by_dev = ar_rows.groupby(col_device)[col_total].sum().sort_values(ascending=True)
    #     chosen_dev = ar_by_dev.index[0]
    #     ar_latency = ar_rows[ar_rows[col_device] == chosen_dev][col_min].min()
    #     df_comp = df[df[col_device] == chosen_dev]
    # else:
    ar_latency = float(ar_rows[col_min].min())
    df_comp = df

    # Compute kernel groups and durations using Avg (ns)
    # Define index windows and associated kernel name patterns.
    # When the overlap window merges (e.g., (0,1)+(2,3) -> (0,3)), we will include
    # all patterns whose index ranges intersect with the requested window.
    groups = [
        ("pre", (0, 1), ['triton_poi_fused_add_0', 'triton_poi_fused_native_dropout_1', 'rmsnorm_fwd_general_kernel']),
        ("fc1", (2, 3), ['nvjet_tst_192x192_64x4_1x1_h_bz_coopB_TNN']),
        ("act", (4, 4), ['triton_poi_fused_mul_silu_0']),
        ("fc2", (5, 6), ['nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN']),
    ]

    def sum_avg_for(names):
        if not names:
            return 0.0
        mask = False
        for nm in names:
            # Use substring match to be robust to name variations
            cur = df_comp[col_kernel].str.contains(nm, na=False)
            mask = cur if isinstance(mask, bool) else (mask | cur)
        if isinstance(mask, bool):
            return 0.0
        matched = df_comp.loc[mask, col_avg]
        if matched.empty:
            return 0.0
        return float(matched.sum())

    group_durations = []
    for key, (gstart, gend), names in groups:
        group_durations.append({
            'key': key,
            'start_idx': gstart,
            'end_idx': gend,
            'names': names,
            'duration': sum_avg_for(names),
        })

    ov_start, ov_end = overlap_window

    if ov_start == -1 and ov_end == -1:
        pre_time = 0.0
        overlap_time = 0.0
        post_time = float(sum(g['duration'] for g in group_durations))
    else:
        pre_time = float(sum(g['duration'] for g in group_durations if g['end_idx'] < ov_start))
        # Include all groups that intersect with the overlap window, accounting for merged ranges like (0,3)
        overlap_time = float(sum(g['duration'] for g in group_durations if not (g['end_idx'] < ov_start or g['start_idx'] > ov_end)))
        post_time = float(sum(g['duration'] for g in group_durations if g['start_idx'] > ov_end))

    overlap_start = pre_time
    non_overlap_start = max(overlap_start + overlap_time, pre_time + ar_latency)
    # Total time from profiling: pre + max(overlap, AR) + post
    total_time = pre_time + max(overlap_time, ar_latency) + post_time
    # Overlap fraction
    overlap_fraction = 0.0 if (ov_start == -1 and ov_end == -1) or total_time <= 0 else min(ar_latency, overlap_time) / total_time

    if draw:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4))
        ax1.set_ylabel('Stream 1')
        ax2.set_ylabel('Stream 2')
        ax1.set_xlim(0, total_time * 1.1 if total_time > 0 else 1)
        ax2.set_xlim(0, total_time * 1.1 if total_time > 0 else 1)
        ax1.set_ylim(0, 0.5)
        ax2.set_ylim(0, 0.5)

        if pre_time > 0:
            ax1.broken_barh([(0, pre_time)], (0, 0.5), facecolors='orange', label='Pre-Compute')
        if ar_latency > 0:
            ax1.broken_barh([(overlap_start, ar_latency)], (0, 0.5), facecolors='blue', label='AllReduce')
        if overlap_time > 0:
            ax2.broken_barh([(overlap_start, overlap_time)], (0, 0.5), facecolors='green', label='Overlapped Compute')
        if post_time > 0:
            ax1.broken_barh([(non_overlap_start, post_time)], (0, 0.5), facecolors='red', label='Post-Compute')

        ax1.axvline(x=overlap_start, color='gray', linestyle='--')
        ax2.axvline(x=overlap_start, color='gray', linestyle='--')
        ax1.axvline(x=non_overlap_start, color='gray', linestyle='--')
        ax2.axvline(x=non_overlap_start, color='gray', linestyle='--')

        ax1.legend(loc='upper right')
        ax2.legend(loc='upper right')
        if title_suffix is None:
            title_suffix = os.path.basename(csv_path)
        plt.suptitle(f"Overlap Timeline | {title_suffix}")
        plt.xlabel('Time (ns) - relative')
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png)
        plt.close()
    stats = {
        'pre_time_ns': pre_time,
        'overlap_time_ns': overlap_time,
        'post_time_ns': post_time,
        'ar_latency_ns': ar_latency,
        'total_time_ns': total_time,
        'overlap_fraction': overlap_fraction,
    }
    return True, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frequency", "-f", type=str, default="default")
    parser.add_argument("--world_size", "-w", type=int, default=2)
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--seq_len", "-s", type=int, default=4096)
    parser.add_argument("--input_dir", type=str, default="", help="If empty, derives from parameters: profile_result/tp<w>-bs<b>-seq<s>/<f>")
    parser.add_argument("--output_dir", type=str, default="", help="If empty, derives from parameters: results/tp<w>-bs<b>-seq<s>/<f>")
    parser.add_argument("--no_time_filter", action="store_true", help="Disable measured time filter (< time_threshold)")
    parser.add_argument("--time_threshold", type=float, default=0.006, help="Measured time threshold for plotting (seconds)")
    parser.add_argument("--no_idle_filter", action="store_true", help="Disable sm_idle_ratio filter from summary csv")
    parser.add_argument("--idle_threshold", type=float, default=0.1, help="sm_idle_ratio threshold for filtering")
    args = parser.parse_args()

    frequency = args.frequency
    if args.input_dir and len(args.input_dir.strip()) > 0:
        input_dir = args.input_dir
    else:
        input_dir = os.path.join(
            "profile_result",
            f"tp{args.world_size}-bs{args.batch_size}-seq{args.seq_len}",
            frequency,
        )

    if args.output_dir and len(args.output_dir.strip()) > 0:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            "results",
            f"tp{args.world_size}-bs{args.batch_size}-seq{args.seq_len}",
            frequency,
        )
    os.makedirs(output_dir, exist_ok=True)

    runs = find_all_runs(input_dir)
    if not runs:
        print(f"No runs found in {input_dir}")
        return

    # If there is at least one existing timeline figure, skip drawing timelines
    existing_timelines = [n for n in os.listdir(output_dir) if n.endswith('.timeline.png')]
    skip_timeline_draw = len(existing_timelines) > 0

    scatter_points = []
    for run in runs:
        base = run["basename"]
        in_csv = run["csv"]
        out_png = os.path.join(output_dir, f"{base}.timeline.png")
        title = f"({run['overlap_start']},{run['overlap_end']}) | SM {run['sm_num']}, Block {run['block_size']}"
        ok, stats = draw_timeline_from_csv(
            in_csv,
            out_png,
            (run['overlap_start'], run['overlap_end']),
            title_suffix=title,
            draw=(not skip_timeline_draw),
        )
        if ok and (not skip_timeline_draw):
            print(f"[✓] Timeline written: {out_png}")
        if ok and stats is not None:
            scatter_points.append({
                'overlap_start': run['overlap_start'],
                'overlap_end': run['overlap_end'],
                'sm_num': run['sm_num'],
                'block_size': run['block_size'],
                **stats,
            })
        # exit()

    # Write raw stats CSV and build scatter plots using energy_results.csv
    logs_dir = os.path.join(
        "logs",
        f"tp{args.world_size}-bs{args.batch_size}-seq{args.seq_len}",
        frequency,
    )
    if scatter_points:
        try:
            os.makedirs(logs_dir, exist_ok=True)
            stats_df = pd.DataFrame(scatter_points)
            stats_csv_path = os.path.join(logs_dir, "overlap_stats.csv")
            stats_df.to_csv(stats_csv_path, index=False)
            print(f"[✓] Stats written: {stats_csv_path}")
        except Exception as e:
            print(f"[!] Failed to write stats CSV: {e}")
    energy_csv = os.path.join(logs_dir, "energy_results.csv")
    if os.path.exists(energy_csv) and scatter_points:
        # try:
        energy_df = pd.read_csv(energy_csv)
        req_cols = [
            "overlap_start",
            "overlap_end",
            "comm_sm_number",
            "comm_block_size",
            "0:time (s)",
            "0:total energy (J)",
        ]
        missing = [c for c in req_cols if c not in energy_df.columns]
        if missing:
            print(f"energy_results.csv missing columns: {missing}")
        else:
            pts = pd.DataFrame(scatter_points)
            # Optional filter from summary_mlp_<frequency>.csv
            summary_csv = os.path.join(logs_dir, f"summary_mlp_{frequency}.csv")
            if (not args.no_idle_filter) and os.path.exists(summary_csv):
                try:
                    summary_df = pd.read_csv(summary_csv)
                    need_cols = ["overlap_start", "overlap_end", "sm_num", "block_size", "sm_idle_ratio"]
                    if all(c in summary_df.columns for c in need_cols):
                        pts = pts.merge(
                            summary_df[need_cols],
                            how="left",
                            on=["overlap_start", "overlap_end", "sm_num", "block_size"],
                        )
                        pts = pts[pts["sm_idle_ratio"].astype(float) < float(args.idle_threshold)]
                    else:
                        print("summary_mlp missing required columns; skipping sm_idle_ratio filter")
                except Exception as e:
                    print(f"Failed to read summary_mlp csv: {e}")

            pts["comm_sm_number"] = pts["sm_num"]
            pts["comm_block_size"] = pts["block_size"]
            merged = pts.merge(
                energy_df[req_cols],
                how="left",
                on=["overlap_start", "overlap_end", "comm_sm_number", "comm_block_size"],
            )
            # Scatter: overlap_fraction vs measured time (s)
            if not merged.empty:
                plt.figure(figsize=(6, 4))
                x = merged["overlap_fraction"].astype(float)
                y = merged["0:time (s)"].astype(float)
                if args.no_time_filter:
                    mask = pd.Series([True] * len(y))
                else:
                    mask = y < float(args.time_threshold)
                x_plot = x[mask]
                y_plot = y[mask]
                plt.scatter(x_plot, y_plot, s=20, alpha=0.8)
                plt.xlabel("Overlap fraction")
                plt.ylabel("Measured time (s)")
                plt.title("Overlap fraction vs measured time")
                os.makedirs(logs_dir, exist_ok=True)
                scatter_png = os.path.join(logs_dir, "overlap_fraction_vs_time.png")
                plt.tight_layout()
                plt.savefig(scatter_png)
                plt.close()
            # Save merged CSV
            merged_out = os.path.join(logs_dir, "overlap_fraction_vs_time.csv")
            merged.to_csv(merged_out, index=False)
            print(f"[✓] Scatter written: {scatter_png}")

            # Scatter: overlap_fraction vs total energy (J)
            if not merged.empty:
                plt.figure(figsize=(6, 4))
                e = merged["0:total energy (J)"].astype(float)
                plt.scatter(x_plot, e[mask], s=20, alpha=0.8)
                plt.xlabel("Overlap fraction")
                plt.ylabel("Total energy (J)")
                plt.title("Overlap fraction vs total energy")
                energy_scatter_png = os.path.join(logs_dir, "overlap_fraction_vs_energy.png")
                plt.tight_layout()
                plt.savefig(energy_scatter_png)
                plt.close()
            energy_csv_out = os.path.join(logs_dir, "overlap_fraction_vs_energy.csv")
            merged.to_csv(energy_csv_out, index=False)
            print(f"[✓] Scatter written: {energy_scatter_png}")
        # except Exception as e:
        #     print(f"[!] Failed to build scatter: {e}")


if __name__ == "__main__":
    main()


