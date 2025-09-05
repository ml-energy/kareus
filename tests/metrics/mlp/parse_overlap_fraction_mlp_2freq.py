import os
import argparse
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Matplotlib settings for research-friendly, deterministic outputs
# - Avoid Type 3 fonts in PDF/PS (use TrueType)
# - Ensure SVG text is preserved (not converted to paths)
# - Make SVG deterministic via fixed hashsalt
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["svg.hashsalt"] = "42"
# Typography: prefer Aptos, with sensible fallbacks; bump default sizes
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Aptos", "Arial", "DejaVu Sans", "Liberation Sans"]
mpl.rcParams["font.size"] = 14
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["xtick.labelsize"] = 13
mpl.rcParams["ytick.labelsize"] = 13
mpl.rcParams["legend.fontsize"] = 14
mpl.rcParams["figure.titlesize"] = 18


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
    # SM idle ratio computed from timeline (no external file):
    # fraction of AR not hidden by overlap, normalized by total iteration time
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
        # Save PNG as before; additionally export PDF/SVG deterministically
        plt.savefig(out_png)
        base_no_ext = os.path.splitext(out_png)[0]
        plt.savefig(base_no_ext + ".pdf", metadata={"CreationDate": None})
        plt.savefig(base_no_ext + ".svg", metadata={"Date": None})
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


def process_frequency(frequency, world_size, batch_size, seq_len, gpu_type,
                      input_dir_arg, output_dir_arg,
                      no_time_filter, time_threshold):
    if input_dir_arg and len(input_dir_arg.strip()) > 0:
        input_dir = input_dir_arg
    else:
        input_dir = os.path.join(
            "profile_result",
            f"tp{world_size}-bs{batch_size}-seq{seq_len}",
            frequency,
        )

    if output_dir_arg and len(output_dir_arg.strip()) > 0:
        output_dir = output_dir_arg
    else:
        output_dir = os.path.join(
            "results",
            f"tp{world_size}-bs{batch_size}-seq{seq_len}",
            frequency,
        )
    os.makedirs(output_dir, exist_ok=True)

    runs = find_all_runs(input_dir)
    if not runs:
        print(f"No runs found in {input_dir}")
        return None, None

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
            gpu_type=gpu_type,
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

    # Write raw stats CSV and build scatter plots using energy_results.csv
    logs_dir = os.path.join(
        "logs",
        f"tp{world_size}-bs{batch_size}-seq{seq_len}",
        frequency,
    )
    energy_csv = os.path.join(logs_dir, "energy_results.csv")
    if os.path.exists(energy_csv) and scatter_points:
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
            return None, logs_dir
        pts = pd.DataFrame(scatter_points)
        # sm_idle_ratio is computed from the timeline stats; no external merge needed

        # Rename energy_df columns to canonical names
        energy_df = energy_df.rename(columns={
            "comm_sm_number": "sm_num",
            "comm_block_size": "block_size",
            """0:time (s)""": "time_s",
            """0:total energy (J)""": "total_energy_J",
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

        # Scatter: per-frequency time
        if not merged.empty:
            x = merged["overlap_fraction"].astype(float)
            y = merged["time_s"].astype(float)
            mask = pd.Series([True] * len(y)) if no_time_filter else (y < float(time_threshold))
            x_plot = x[mask]
            y_plot = y[mask]
            plt.figure(figsize=(6, 4))
            if "sm_idle_ratio" in merged.columns:
                c_vals = merged["sm_idle_ratio"].astype(float)[mask]
                sc = plt.scatter(x_plot, y_plot, s=20, alpha=0.8, c=c_vals, cmap='viridis')
                cbar = plt.colorbar(sc)
                cbar.set_label('Exposed Communication Fraction')
            else:
                plt.scatter(x_plot, y_plot, s=20, alpha=0.8)
            plt.xlabel("Overlap Fraction")
            plt.ylabel("Time (s)")
            os.makedirs(logs_dir, exist_ok=True)
            scatter_png = os.path.join(logs_dir, "overlap_fraction_vs_time.pdf")
            plt.tight_layout()
            plt.savefig(scatter_png)
            # Deterministic PDF/SVG exports
            base_no_ext = os.path.splitext(scatter_png)[0]
            plt.savefig(base_no_ext + ".pdf", metadata={"CreationDate": None})
            plt.savefig(base_no_ext + ".svg", metadata={"Date": None})
            plt.close()

            # Scatter: per-frequency energy
            plt.figure(figsize=(6, 4))
            e = merged["total_energy_J"].astype(float)
            if "sm_idle_ratio" in merged.columns:
                c_vals_e = merged["sm_idle_ratio"].astype(float)[mask]
                sc2 = plt.scatter(x_plot, e[mask], s=20, alpha=0.8, c=c_vals_e, cmap='viridis')
                cbar2 = plt.colorbar(sc2)
                cbar2.set_label('Exposed Communication Fraction')
            else:
                plt.scatter(x_plot, e[mask], s=20, alpha=0.8)
            plt.xlabel("Overlap Fraction")
            plt.ylabel("Energy (J)")
            energy_scatter_png = os.path.join(logs_dir, "overlap_fraction_vs_energy.pdf")
            plt.tight_layout()
            plt.savefig(energy_scatter_png)
            # Deterministic PDF/SVG exports
            base_no_ext = os.path.splitext(energy_scatter_png)[0]
            plt.savefig(base_no_ext + ".pdf", metadata={"CreationDate": None})
            plt.savefig(base_no_ext + ".svg", metadata={"Date": None})
            plt.close()
            energy_csv_out = os.path.join(logs_dir, "overlap_fraction.csv")
            merged.to_csv(energy_csv_out, index=False)
            print(f"[✓] Scatter written: {energy_scatter_png}")
        return merged, logs_dir
    return None, logs_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frequency", "-f", type=str, default="1400")
    parser.add_argument("--frequency2", "-f2", type=str, default="1100", help="Optional second frequency to overlay")
    parser.add_argument("--world_size", "-w", type=int, default=2)
    parser.add_argument("--batch_size", "-b", type=int, default=8)
    parser.add_argument("--seq_len", "-s", type=int, default=4096)
    parser.add_argument("--gpu_type", type=str, default="ampere", help="GPU type: hopper or ampere")
    parser.add_argument("--input_dir", type=str, default="", help="If empty, derives from parameters: profile_result/tp<w>-bs<b>-seq<s>/<f>")
    parser.add_argument("--output_dir", type=str, default="", help="If empty, derives from parameters: results/tp<w>-bs<b>-seq<s>/<f>")
    parser.add_argument("--input_dir2", type=str, default="", help="Optional explicit input dir for frequency2")
    parser.add_argument("--output_dir2", type=str, default="", help="Optional explicit output dir for frequency2")
    parser.add_argument("--no_time_filter", action="store_true", help="Disable measured time filter (< time_threshold)")
    parser.add_argument("--time_threshold", type=float, default=0.0075, help="Measured time threshold for plotting (seconds)")
    parser.add_argument("--no_time_filter2", action="store_true", help="Disable measured time filter for frequency2")
    parser.add_argument("--time_threshold2", type=float, default=0.0085, help="Measured time threshold for plotting for frequency2 (seconds)")
    args = parser.parse_args()

    # Use new helper for frequency 1
    merged1, logs_dir1 = process_frequency(
        frequency=args.frequency,
        world_size=args.world_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        gpu_type=args.gpu_type,
        input_dir_arg=args.input_dir,
        output_dir_arg=args.output_dir,
        no_time_filter=args.no_time_filter,
        time_threshold=args.time_threshold,
    )

    # Optional frequency 2
    frequency2 = (args.frequency2 or "").strip()
    merged2 = None
    logs_dir2 = None
    if frequency2:
        merged2, logs_dir2 = process_frequency(
            frequency=frequency2,
            world_size=args.world_size,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            gpu_type=args.gpu_type,
            input_dir_arg=args.input_dir2,
            output_dir_arg=args.output_dir2,
            no_time_filter=args.no_time_filter2,
            time_threshold=args.time_threshold2,
        )

    # Draw combined overlays if both datasets exist
    if merged1 is not None and frequency2 and merged2 is not None:
        logs_dir_base = os.path.join(
            "logs",
            f"tp{args.world_size}-bs{args.batch_size}-seq{args.seq_len}",
        )
        combined_dir = os.path.join(logs_dir_base, "combined")
        os.makedirs(combined_dir, exist_ok=True)

        # Time overlay
        plt.figure(figsize=(6, 4))
        x1 = merged1["overlap_fraction"].astype(float)
        y1 = merged1["time_s"].astype(float)
        mask1 = pd.Series([True] * len(y1)) if args.no_time_filter else (y1 < float(args.time_threshold))
        x2 = merged2["overlap_fraction"].astype(float)
        y2 = merged2["time_s"].astype(float)
        mask2 = pd.Series([True] * len(y2)) if args.no_time_filter2 else (y2 < float(args.time_threshold2))

        has_c1 = "sm_idle_ratio" in merged1.columns
        has_c2 = "sm_idle_ratio" in merged2.columns
        if has_c1 or has_c2:
            vals = []
            if has_c1:
                vals.extend(merged1["sm_idle_ratio"].astype(float)[mask1].dropna().tolist())
            if has_c2:
                vals.extend(merged2["sm_idle_ratio"].astype(float)[mask2].dropna().tolist())
            if len(vals) > 0:
                vmin = min(vals)
                vmax = max(vals)
            else:
                vmin, vmax = None, None
            if has_c1:
                plt.scatter(x1[mask1], y1[mask1], s=26, alpha=0.85, marker='o',
                            c=merged1["sm_idle_ratio"].astype(float)[mask1], cmap='viridis', vmin=vmin, vmax=vmax,
                            label="1410 MHz")
            else:
                plt.scatter(x1[mask1], y1[mask1], s=26, alpha=0.85, marker='o', color='tab:blue', label=args.frequency)
            if has_c2:
                plt.scatter(x2[mask2], y2[mask2], s=26, alpha=0.85, marker='^',
                            c=merged2["sm_idle_ratio"].astype(float)[mask2], cmap='viridis', vmin=vmin, vmax=vmax,
                            label="_nolegend_")
            else:
                plt.scatter(x2[mask2], y2[mask2], s=26, alpha=0.85, marker='^', color='tab:orange', label='_nolegend_')
        else:
            plt.scatter(x1[mask1], y1[mask1], s=26, alpha=0.85, marker='o', color='tab:blue', label=args.frequency)
            plt.scatter(x2[mask2], y2[mask2], s=26, alpha=0.85, marker='^', color='tab:orange', label='_nolegend_')

        plt.xlabel("Overlap Fraction")
        plt.ylabel("Time (s)")
        plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.18), frameon=False)
        out_time_overlay = os.path.join(combined_dir, f"mlp_overlap_fraction_vs_time__{args.frequency}_vs_{frequency2}.pdf")
        plt.tight_layout()
        plt.savefig(out_time_overlay, bbox_inches='tight')
        # Deterministic PDF/SVG exports
        base_no_ext = os.path.splitext(out_time_overlay)[0]
        plt.savefig(base_no_ext + ".pdf", metadata={"CreationDate": None}, bbox_inches='tight')
        plt.savefig(base_no_ext + ".svg", metadata={"Date": None}, bbox_inches='tight')
        plt.close()

        # Energy overlay
        plt.figure(figsize=(6, 4))
        e1 = merged1["total_energy_J"].astype(float)
        e2 = merged2["total_energy_J"].astype(float)
        has_c1 = "sm_idle_ratio" in merged1.columns
        has_c2 = "sm_idle_ratio" in merged2.columns
        cbar_added = False
        if has_c1 or has_c2:
            vals = []
            if has_c1:
                vals.extend(merged1["sm_idle_ratio"].astype(float)[mask1].dropna().tolist())
            if has_c2:
                vals.extend(merged2["sm_idle_ratio"].astype(float)[mask2].dropna().tolist())
            if len(vals) > 0:
                vmin = min(vals)
                vmax = max(vals)
            else:
                vmin, vmax = None, None
            if has_c1:
                sc1 = plt.scatter(x1[mask1], e1[mask1], s=26, alpha=0.85, marker='o',
                                  c=merged1["sm_idle_ratio"].astype(float)[mask1], cmap='viridis', vmin=vmin, vmax=vmax,
                                  label='_nolegend_')
                cbar = plt.colorbar(sc1)
                cbar.set_label('Exposed Communication Fraction')
                cbar_added = True
            else:
                plt.scatter(x1[mask1], e1[mask1], s=26, alpha=0.85, marker='o', color='tab:blue', label='_nolegend_')
            if has_c2:
                plt.scatter(x2[mask2], e2[mask2], s=26, alpha=0.85, marker='^',
                            c=merged2["sm_idle_ratio"].astype(float)[mask2], cmap='viridis', vmin=vmin, vmax=vmax,
                            label='1100 MHz')
                if not cbar_added:
                    sc2 = plt.scatter([], [], c=[], cmap='viridis')
                    cbar = plt.colorbar(sc2)
                    cbar.set_label('Exposed Communication Fraction')
            else:
                plt.scatter(x2[mask2], e2[mask2], s=26, alpha=0.85, marker='^', color='tab:orange', label='1100 MHz')
        else:
            plt.scatter(x1[mask1], e1[mask1], s=26, alpha=0.85, marker='o', color='tab:blue', label='_nolegend_')
            plt.scatter(x2[mask2], e2[mask2], s=26, alpha=0.85, marker='^', color='tab:orange', label='1100 MHz')

        plt.xlabel("Overlap Fraction")
        plt.ylabel("Energy (J)")
        plt.legend(loc='upper left', bbox_to_anchor=(0.0, 1.18), frameon=False)
        out_energy_overlay = os.path.join(combined_dir, f"mlp_overlap_fraction_vs_energy__{args.frequency}_vs_{frequency2}.pdf")
        plt.tight_layout()
        plt.savefig(out_energy_overlay, bbox_inches='tight')
        # Deterministic PDF/SVG exports
        base_no_ext = os.path.splitext(out_energy_overlay)[0]
        plt.savefig(base_no_ext + ".pdf", metadata={"CreationDate": None}, bbox_inches='tight')
        plt.savefig(base_no_ext + ".svg", metadata={"Date": None}, bbox_inches='tight')
        plt.close()

        # Combined CSV export
        combined = merged1.copy()
        combined["frequency"] = args.frequency
        tmp = merged2.copy()
        tmp["frequency"] = frequency2
        combined = pd.concat([combined, tmp], ignore_index=True)
        combined.to_csv(os.path.join(combined_dir, f"overlap_fraction__{args.frequency}_vs_{frequency2}.csv"), index=False)
        print(f"[✓] Combined overlays written: {out_time_overlay}, {out_energy_overlay}")


if __name__ == "__main__":
    main()


