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
        suffix = ".csv_cuda_gpu_kern_sum.csv"
        if not name.endswith(suffix):
            continue
        base = name[:-len(suffix)]
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
        csv_path = os.path.join(results_dir, name)
        runs.append({
            "csv": csv_path,
            "png": os.path.join(results_dir, f"{base}.timeline.png"),
            "overlap_start": overlap_start,
            "overlap_end": overlap_end,
            "sm_num": sm_num,
            "block_size": block_size,
            "basename": base,
        })
    return sorted(runs, key=lambda x: x["basename"])    


def draw_timeline_from_csv(csv_path, out_png, overlap_window, title_suffix=None, draw=True, gpu_type="hopper"):
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
    col_max = cols.get('max (ns)'.lower())
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
    # Determine FC kernel name patterns by GPU type
    if (gpu_type or "").lower() == "ampere":
        fc1_names = ['ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn']
        fc2_names = ['ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn']
    else:  # hopper (default)
        fc1_names = ['nvjet_tst_192x192_64x4_1x1_h_bz_coopB_TNN']
        fc2_names = ['nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN']

    groups = [
        ("pre", (0, 1), ['triton_poi_fused_add_0', 'triton_poi_fused_native_dropout_1', 'rmsnorm_fwd_general_kernel']),
        ("fc1", (2, 3), fc1_names),
        ("act", (4, 4), ['triton_poi_fused_mul_silu_0']),
        ("fc2", (5, 6), fc2_names),
    ]

    def aggregate_duration(names, mode):
        if not names:
            return 0.0
        mask = False
        for nm in names:
            cur = df_comp[col_kernel].str.contains(nm, na=False)
            mask = cur if isinstance(mask, bool) else (mask | cur)
        if isinstance(mask, bool):
            print(f"No matches found for {names}")
            return 0.0
        matched_rows = df_comp.loc[mask]
        if matched_rows.empty:
            print(f"No matched rows found for {names}")
            return 0.0
        if mode == 'fc1_max':
            return float(matched_rows[col_max].astype(float).sum())
        if mode == 'fc2_min':
            return float(matched_rows[col_min].astype(float).sum())
        # Default: sum of Avg (ns)
        return float(matched_rows[col_avg].astype(float).sum())

    group_durations = []
    for key, (gstart, gend), names in groups:
        if key == 'fc1':
            duration = aggregate_duration(names, 'fc1_max')
        elif key == 'fc2':
            duration = aggregate_duration(names, 'fc2_min')
        else:
            duration = aggregate_duration(names, 'sum_avg')
        group_durations.append({
            'key': key,
            'start_idx': gstart,
            'end_idx': gend,
            'names': names,
            'duration': duration,
        })

    ov_start, ov_end = overlap_window

    # Classify groups into pre/overlap/post based on the requested overlap window
    if ov_start == -1 and ov_end == -1:
        pre_groups = []
        overlap_groups = []
        post_groups = list(group_durations)
    else:
        pre_groups = [g for g in group_durations if g['end_idx'] < ov_start]
        # Include all groups that intersect with the overlap window, accounting for merged ranges like (0,3)
        overlap_groups = [g for g in group_durations if not (g['end_idx'] < ov_start or g['start_idx'] > ov_end)]
        post_groups = [g for g in group_durations if g['start_idx'] > ov_end]

    pre_time = float(sum(g['duration'] for g in pre_groups))
    overlap_time = float(sum(g['duration'] for g in overlap_groups))
    post_time = float(sum(g['duration'] for g in post_groups))

    overlap_start = pre_time
    non_overlap_start = max(overlap_start + overlap_time, pre_time + ar_latency)
    # Total time from profiling: pre + max(overlap, AR) + post
    total_time = pre_time + max(overlap_time, ar_latency) + post_time
    # Overlap fraction
    overlap_fraction = 0.0 if (ov_start == -1 and ov_end == -1) or total_time <= 0 else min(ar_latency, overlap_time) / total_time
    # SM idle ratio: (AR time - overlap time) floored at 0, normalized by total time
    sm_idle_ratio = 0.0 if total_time <= 0 else max(ar_latency - overlap_time, 0.0) / total_time

    if draw:
        # Convert all times from ns to ms for plotting
        ns_to_ms = 1e-6
        pre_time_ms = pre_time * ns_to_ms
        overlap_time_ms = overlap_time * ns_to_ms
        post_time_ms = post_time * ns_to_ms
        ar_latency_ms = ar_latency * ns_to_ms
        total_time_ms = total_time * ns_to_ms
        overlap_start_ms = overlap_start * ns_to_ms
        non_overlap_start_ms = non_overlap_start * ns_to_ms

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4))
        ax1.set_ylabel('Stream 1')
        ax2.set_ylabel('Stream 2')
        ax1.set_xlim(0, total_time_ms * 1.1 if total_time_ms > 0 else 1)
        ax2.set_xlim(0, total_time_ms * 1.1 if total_time_ms > 0 else 1)
        ax1.set_ylim(0, 0.5)
        ax2.set_ylim(0, 0.5)

        # Friendly names for each kernel group
        group_labels = {
            'pre': 'RMSNorm',
            'fc1': 'Linear1',
            'act': 'SiLU',
            'fc2': 'Linear2',
        }

        # Distinct colors for each kernel group
        group_colors = {
            'pre': '#FFA500',   # orange
            'fc1': '#8A2BE2',   # blueviolet
            'act': '#20B2AA',   # lightseagreen
            'fc2': '#DC143C',   # crimson
        }

        # Keep track of labels already added to legends to avoid duplicates
        used_labels_ax1 = set()
        used_labels_ax2 = set()

        # Draw pre-compute groups on Stream 1 before overlap_start
        x_pre = 0.0
        for g in pre_groups:
            dur = float(g['duration'])
            if dur <= 0:
                continue
            label = group_labels.get(g['key'], g['key'])
            color = group_colors.get(g['key'], '#808080')
            dur_ms = dur * ns_to_ms
            ax1.broken_barh([(x_pre, dur_ms)], (0, 0.5), facecolors=color, label=None)
            used_labels_ax1.add(label)
            # Annotate with group name
            ax1.text(x_pre + dur_ms / 2.0, 0.25, label, ha='center', va='center', fontsize=12, color='black')
            x_pre += dur_ms

        # Draw AllReduce on Stream 2 during the overlap window
        if ar_latency > 0:
            ar_label = 'AllReduce'
            ax2.broken_barh([(overlap_start_ms, ar_latency_ms)], (0, 0.5), facecolors='blue', label=None)
            # Annotate AllReduce text inside the bar
            ax2.text(overlap_start_ms + ar_latency_ms / 2.0, 0.25, ar_label, ha='center', va='center', fontsize=12, color='black')
            used_labels_ax2.add(ar_label)

        # Draw overlapped compute groups on Stream 1 between overlap_start and overlap_end (time)
        x_ov = overlap_start_ms
        for g in overlap_groups:
            dur = float(g['duration'])
            if dur <= 0:
                continue
            label = group_labels.get(g['key'], g['key'])
            color = group_colors.get(g['key'], '#808080')
            dur_ms = dur * ns_to_ms
            ax1.broken_barh([(x_ov, dur_ms)], (0, 0.5), facecolors=color, label=None)
            used_labels_ax1.add(label)
            ax1.text(x_ov + dur_ms / 2.0, 0.25, label, ha='center', va='center', fontsize=12, color='black')
            x_ov += dur_ms

        # Draw post-compute groups on Stream 1 after overlap_end (time)
        x_post = non_overlap_start_ms
        for g in post_groups:
            dur = float(g['duration'])
            if dur <= 0:
                continue
            label = group_labels.get(g['key'], g['key'])
            color = group_colors.get(g['key'], '#808080')
            dur_ms = dur * ns_to_ms
            ax1.broken_barh([(x_post, dur_ms)], (0, 0.5), facecolors=color, label=None)
            used_labels_ax1.add(label)
            ax1.text(x_post + dur_ms / 2.0, 0.25, label, ha='center', va='center', fontsize=12, color='black')
            x_post += dur_ms

        ax1.axvline(x=overlap_start_ms, color='gray', linestyle='--')
        ax2.axvline(x=overlap_start_ms, color='gray', linestyle='--')
        ax1.axvline(x=non_overlap_start_ms, color='gray', linestyle='--')
        ax2.axvline(x=non_overlap_start_ms, color='gray', linestyle='--')

        # Legends removed as requested
        if title_suffix is None:
            title_suffix = os.path.basename(csv_path)
        plt.suptitle(f"Overlap Timeline | {title_suffix}")
        plt.xlabel('Time (ms)')
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
        'sm_idle_ratio': sm_idle_ratio,
    }
    return True, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frequency", "-f", type=str, default="default")
    parser.add_argument("--world_size", "-w", type=int, default=2)
    parser.add_argument("--batch_size", "-b", type=int, default=8)
    parser.add_argument("--seq_len", "-s", type=int, default=4096)
    parser.add_argument("--gpu_type", type=str, default="ampere", help="GPU type: hopper or ampere")
    parser.add_argument("--input_dir", type=str, default="", help="If empty, derives from parameters: profile_result/tp<w>-bs<b>-seq<s>/<f>")
    parser.add_argument("--output_dir", type=str, default="", help="If empty, derives from parameters: results/tp<w>-bs<b>-seq<s>/<f>")
    parser.add_argument("--no_time_filter", action="store_true", help="Disable measured time filter (< time_threshold)")
    parser.add_argument("--time_threshold", type=float, default=0.008, help="Measured time threshold for plotting (seconds)")
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
    # skip_timeline_draw = False

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
            gpu_type=args.gpu_type,
        )
        if ok and (not skip_timeline_draw):
            print(f"[✓] Timeline written: {out_png}")
        if ok and stats is not None and run['sm_num'] <= 20:
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
    energy_csv = os.path.join(logs_dir, "energy_results.csv")
    if os.path.exists(energy_csv) and scatter_points:
        # try:
        energy_df = pd.read_csv(energy_csv)
        # Require original column names, then rename to canonical ones
        req_cols_original = [
            "overlap_start",
            "overlap_end",
            "comm_sm_number",
            "comm_block_size",
            "0:time (s)",
            "0:total energy (J)",
        ]
        missing = [c for c in req_cols_original if c not in energy_df.columns]
        if missing:
            print(f"energy_results.csv missing columns: {missing}")
        else:
            pts = pd.DataFrame(scatter_points)
            # sm_idle_ratio is computed from durations; no external merge required

            # Rename energy_df columns to canonical names
            energy_df = energy_df.rename(columns={
                "comm_sm_number": "sm_num",
                "comm_block_size": "block_size",
                "0:time (s)": "time_s",
                "0:total energy (J)": "total_energy_J",
            })
            req_cols = [
                "overlap_start",
                "overlap_end",
                "sm_num",
                "block_size",
                "time_s",
                "total_energy_J",
            ]
            merged = pts.merge(
                energy_df[req_cols],
                how="left",
                on=["overlap_start", "overlap_end", "sm_num", "block_size"],
            )
            # Scatter: overlap_fraction vs measured time (s)
            if not merged.empty:
                plt.figure(figsize=(6, 4))
                x = merged["overlap_fraction"].astype(float)
                y = merged["time_s"].astype(float)
                if args.no_time_filter:
                    mask = pd.Series([True] * len(y))
                else:
                    mask = y < float(args.time_threshold)
                x_plot = x[mask]
                y_plot = y[mask]
                # print(f"x_plot: {x_plot}, y_plot: {y_plot}")
                if "sm_idle_ratio" in merged.columns:
                    c_vals = merged["sm_idle_ratio"].astype(float)[mask]
                    sc = plt.scatter(x_plot, y_plot, s=20, alpha=0.8, c=c_vals, cmap='viridis')
                    cbar = plt.colorbar(sc)
                    cbar.set_label('SM idle ratio')
                else:
                    plt.scatter(x_plot, y_plot, s=20, alpha=0.8)
                plt.xlabel("Overlap fraction")
                plt.ylabel("Measured time (s)")
                plt.title("Overlap fraction vs measured time")
                os.makedirs(logs_dir, exist_ok=True)
                scatter_png = os.path.join(logs_dir, "overlap_fraction_vs_time.png")
                plt.tight_layout()
                plt.savefig(scatter_png)
                plt.close()

            # Scatter: overlap_fraction vs total energy (J)
            if not merged.empty:
                plt.figure(figsize=(6, 4))
                e = merged["total_energy_J"].astype(float)
                if "sm_idle_ratio" in merged.columns:
                    c_vals_e = merged["sm_idle_ratio"].astype(float)[mask]
                    # print(f"x_plot: {x_plot}, e: {e[mask]}")
                    sc2 = plt.scatter(x_plot, e[mask], s=20, alpha=0.8, c=c_vals_e, cmap='viridis')
                    cbar2 = plt.colorbar(sc2)
                    cbar2.set_label('SM idle ratio')
                else:
                    plt.scatter(x_plot, e[mask], s=20, alpha=0.8)
                plt.xlabel("Overlap fraction")
                plt.ylabel("Total energy (J)")
                plt.title("Overlap fraction vs total energy")
                energy_scatter_png = os.path.join(logs_dir, "overlap_fraction_vs_energy.png")
                plt.tight_layout()
                plt.savefig(energy_scatter_png)
                plt.close()
            energy_csv_out = os.path.join(logs_dir, "overlap_fraction.csv")
            merged.to_csv(energy_csv_out, index=False)
            print(f"[✓] Scatter written: {energy_scatter_png}")
        # except Exception as e:
        #     print(f"[!] Failed to build scatter: {e}")


if __name__ == "__main__":
    main()


