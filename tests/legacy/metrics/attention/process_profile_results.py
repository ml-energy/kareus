#!/usr/bin/env python3
"""
Process nsys profile results: export to SQLite and draw timelines.
Usage:
    python process_profile_results.py --input_dir profile_result/tp8-bs8-seq4096
    python process_profile_results.py --input_dir profile_result/event/tp8-bs8-seq4096
"""

import os
import sys
import argparse
import sqlite3
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# python process_profile_results.py --input_dir profile_result/tp8-bs8-seq4096


def find_nsys_reports(base_dir):
    """Find all .nsys-rep files recursively in the given directory."""
    reports = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".nsys-rep"):
                full_path = os.path.join(root, f)
                # Parse filename: profile_<overlap_start>_<overlap_end>_<sm_num>_<block_size>.nsys-rep
                basename = f.replace(".nsys-rep", "")
                parts = basename.split("_")
                if len(parts) >= 5 and parts[0] == "profile":
                    try:
                        overlap_start = int(parts[1])
                        overlap_end = int(parts[2])
                        sm_num = int(parts[3])
                        block_size = int(parts[4])
                        # Determine frequency and phase from path
                        rel_path = os.path.relpath(root, base_dir)
                        path_parts = rel_path.split(os.sep)
                        frequency = path_parts[0] if len(path_parts) > 0 else "default"
                        phase = path_parts[1] if len(path_parts) > 1 else "unknown"
                        reports.append({
                            "nsys_rep": full_path,
                            "dirname": root,
                            "basename": basename,
                            "overlap_start": overlap_start,
                            "overlap_end": overlap_end,
                            "sm_num": sm_num,
                            "block_size": block_size,
                            "frequency": frequency,
                            "phase": phase,
                        })
                    except ValueError:
                        print(f"[!] Could not parse: {f}")
                        continue
    return sorted(reports, key=lambda x: (x["frequency"], x["phase"], x["basename"]))


def export_to_sqlite(nsys_rep_path, output_dir, force=False):
    """Export nsys-rep to SQLite and CSV using nsys stats."""
    basename = os.path.basename(nsys_rep_path).replace(".nsys-rep", "")
    sqlite_path = os.path.join(output_dir, f"{basename}.sqlite")
    csv_path = os.path.join(output_dir, f"{basename}.csv")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not force and os.path.exists(sqlite_path):
        print(f"[=] SQLite exists: {sqlite_path}")
        return sqlite_path, csv_path
    
    cmd = [
        "nsys", "stats", nsys_rep_path,
        "--report", "cuda_gpu_kern_sum",
        "--format", "csv",
        "--force-export", "true",
        "--sqlite", sqlite_path,
        "--output", csv_path,
    ]
    print(f"[>] Exporting: {basename}")
    result = subprocess.run(" ".join(cmd), shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[!] nsys stats failed: {result.stderr}")
        return None, None
    
    print(f"[✓] Exported SQLite: {sqlite_path}")
    return sqlite_path, csv_path


def get_kernel_data_from_sqlite(sqlite_path, chosen_gpu=None):
    """Extract kernel execution data from SQLite database."""
    if not os.path.exists(sqlite_path):
        return None
    
    conn = sqlite3.connect(sqlite_path)
    
    # Get all kernels with their timing information
    query = """
    SELECT 
        s.value as kernel_name,
        k.deviceId,
        k.start,
        k.end,
        (k.end - k.start) as duration
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    LEFT JOIN StringIds s ON k.demangledName = s.id
    ORDER BY k.start
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Filter out FillFunctor kernels
    df = df[~df['kernel_name'].str.contains('FillFunctor', na=False)]
    
    # If chosen_gpu specified, filter by device
    if chosen_gpu is not None:
        df = df[df['deviceId'] == chosen_gpu]
    
    return df


def get_kernel_summary_from_csv(csv_path):
    """Read kernel summary from CSV file."""
    suffix = "_cuda_gpu_kern_sum.csv"
    actual_csv = csv_path + suffix if not csv_path.endswith(suffix) else csv_path
    
    if not os.path.exists(actual_csv):
        # Try without suffix
        if os.path.exists(csv_path):
            actual_csv = csv_path
        else:
            return None
    
    df = pd.read_csv(actual_csv)
    return df


def draw_timeline_from_kernel_data(kernel_df, output_png, overlap_window, title=None, gpu_type="ampere", phase="forward"):
    """Draw timeline visualization from kernel execution data.
    
    Args:
        phase: "forward" or "backward" - determines kernel order
               forward: pre, fc1, post, attn, fc2
               backward: pre, fc2, attn, post, fc1
    """
    if kernel_df is None or kernel_df.empty:
        return False, None
    
    # Get summary statistics for each kernel type
    kernel_stats = kernel_df.groupby('kernel_name').agg({
        'duration': ['sum', 'mean', 'min', 'max', 'count'],
        'start': 'min',
        'end': 'max'
    }).reset_index()
    kernel_stats.columns = ['kernel_name', 'total_duration', 'avg_duration', 'min_duration', 
                           'max_duration', 'count', 'first_start', 'last_end']
    
    # Find AllReduce kernel
    ar_rows = kernel_stats[kernel_stats['kernel_name'].str.contains('allreduce', case=False, na=False)]
    if ar_rows.empty:
        ar_latency = 0
    else:
        ar_latency = float(ar_rows['min_duration'].min())
    
    # Define kernel groups for timeline
    if (gpu_type or "").lower() == "ampere":
        fc1_names = ['ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn']
        # fc2_names = ['ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn']
        fc2_names = ['ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn']
    else:  # hopper
        fc1_names = ['nvjet_tst_192x192_64x4_1x1_h_bz_coopB_TNN']
        fc2_names = ['nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN']
    
    # Define groups based on phase (forward vs backward)
    # forward order: pre, fc1, post, attn, fc2
    # backward order: pre, fc2, attn, post, fc1
    if phase == "backward":
        groups = [
            ("pre", (0, 1), ['rmsnorm_bwd_tuned_kernel']),
            ("fc2", (2, 3), ['ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn', 'ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_64x3_nt', 'splitKreduce_kernel']),
            ("attn", (4, 4), ['flash_bwd']),
            ("post", (5, 6), ['at::native::direct_copy_kernel_cuda', 'fused_rope_backward', 'reduce_kernel', 'CatArrayBatchedCopy']),
            ("fc1", (7, 8), ['ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_nn', 'ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_nt']),
        ]
    else:  # forward (default)
        groups = [
            # ("pre", (0, 1), ['triton_poi_fused_add_0', 'triton_poi_fused_native_dropout_1', 'rmsnorm_fwd_general_kernel']),
            ("pre", (0, 1), ['rmsnorm_fwd_tuned_kernel']),
            ("fc1", (2, 3), fc1_names),
            ("post", (4, 5), ['at::native::direct_copy_kernel_cuda', 'fused_rope_forward']),
            ("attn", (6, 6), ['flash_fwd_kernel']),
            ("fc2", (7, 8), fc2_names),
        ]
    
    def aggregate_duration(names, mode='avg'):
        if not names:
            return 0.0
        mask = False
        for nm in names:
            cur = kernel_stats['kernel_name'].str.contains(nm, na=False)
            mask = cur if isinstance(mask, bool) else (mask | cur)
        if isinstance(mask, bool):
            return 0.0
        matched = kernel_stats.loc[mask]
        if matched.empty:
            return 0.0
        if mode == 'max':
            return float(matched['max_duration'].sum())
        elif mode == 'min':
            return float(matched['min_duration'].sum())
        return float(matched['avg_duration'].sum())
    
    group_durations = []
    for key, (gstart, gend), names in groups:
        if key == 'fc1' and phase == 'backward':
            duration = aggregate_duration(names, 'max')
        elif key == 'fc2' and phase == 'backward':
            duration = aggregate_duration(names, 'min')
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
    
    # Classify groups
    if ov_start == -1 and ov_end == -1:
        pre_groups = []
        overlap_groups = []
        post_groups = list(group_durations)
    else:
        pre_groups = [g for g in group_durations if g['end_idx'] < ov_start]
        overlap_groups = [g for g in group_durations if not (g['end_idx'] < ov_start or g['start_idx'] > ov_end)]
        post_groups = [g for g in group_durations if g['start_idx'] > ov_end]
    
    pre_time = float(sum(g['duration'] for g in pre_groups))
    overlap_time = float(sum(g['duration'] for g in overlap_groups))
    post_time = float(sum(g['duration'] for g in post_groups))
    
    overlap_start_time = pre_time
    non_overlap_start = max(overlap_start_time + overlap_time, pre_time + ar_latency)
    total_time = pre_time + max(overlap_time, ar_latency) + post_time
    overlap_fraction = 0.0 if (ov_start == -1 and ov_end == -1) or total_time <= 0 else min(ar_latency, overlap_time) / total_time
    
    # Convert ns to ms
    ns_to_ms = 1e-6
    pre_time_ms = pre_time * ns_to_ms
    overlap_time_ms = overlap_time * ns_to_ms
    post_time_ms = post_time * ns_to_ms
    ar_latency_ms = ar_latency * ns_to_ms
    total_time_ms = total_time * ns_to_ms
    overlap_start_ms = overlap_start_time * ns_to_ms
    non_overlap_start_ms = non_overlap_start * ns_to_ms
    
    # Draw timeline
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4))
    ax1.set_ylabel('Stream 1 (Compute)')
    ax2.set_ylabel('Stream 2 (Comm)')
    ax1.set_xlim(0, total_time_ms * 1.1 if total_time_ms > 0 else 1)
    ax2.set_xlim(0, total_time_ms * 1.1 if total_time_ms > 0 else 1)
    ax1.set_ylim(0, 0.5)
    ax2.set_ylim(0, 0.5)
    
    group_labels = {
        'pre': 'RMSNorm',
        'fc1': 'Linear1',
        'post': 'RoPE',
        'attn': 'Attention',
        'fc2': 'Linear2',
    }
    
    group_colors = {
        'pre': '#FFA500',    # orange
        'fc1': '#8A2BE2',    # blueviolet
        'post': '#20B2AA',   # lightseagreen
        'attn': '#FF69B4',   # hotpink
        'fc2': '#DC143C',    # crimson
    }
    
    # Draw pre-compute groups
    x_pre = 0.0
    for g in pre_groups:
        dur = float(g['duration'])
        if dur <= 0:
            continue
        label = group_labels.get(g['key'], g['key'])
        color = group_colors.get(g['key'], '#808080')
        dur_ms = dur * ns_to_ms
        ax1.broken_barh([(x_pre, dur_ms)], (0, 0.5), facecolors=color)
        ax1.text(x_pre + dur_ms / 2.0, 0.25, label, ha='center', va='center', fontsize=10, color='white')
        x_pre += dur_ms
    
    # Draw AllReduce on Stream 2
    if ar_latency > 0:
        ax2.broken_barh([(overlap_start_ms, ar_latency_ms)], (0, 0.5), facecolors='#4169E1')
        ax2.text(overlap_start_ms + ar_latency_ms / 2.0, 0.25, 'AllReduce', ha='center', va='center', fontsize=10, color='white')
    
    # Draw overlapped compute groups
    x_ov = overlap_start_ms
    for g in overlap_groups:
        dur = float(g['duration'])
        if dur <= 0:
            continue
        label = group_labels.get(g['key'], g['key'])
        color = group_colors.get(g['key'], '#808080')
        dur_ms = dur * ns_to_ms
        ax1.broken_barh([(x_ov, dur_ms)], (0, 0.5), facecolors=color)
        ax1.text(x_ov + dur_ms / 2.0, 0.25, label, ha='center', va='center', fontsize=10, color='white')
        x_ov += dur_ms
    
    # Draw post-compute groups
    x_post = non_overlap_start_ms
    for g in post_groups:
        dur = float(g['duration'])
        if dur <= 0:
            continue
        label = group_labels.get(g['key'], g['key'])
        color = group_colors.get(g['key'], '#808080')
        dur_ms = dur * ns_to_ms
        ax1.broken_barh([(x_post, dur_ms)], (0, 0.5), facecolors=color)
        ax1.text(x_post + dur_ms / 2.0, 0.25, label, ha='center', va='center', fontsize=10, color='white')
        x_post += dur_ms
    
    # Draw vertical lines
    ax1.axvline(x=overlap_start_ms, color='gray', linestyle='--', alpha=0.7)
    ax2.axvline(x=overlap_start_ms, color='gray', linestyle='--', alpha=0.7)
    ax1.axvline(x=non_overlap_start_ms, color='gray', linestyle='--', alpha=0.7)
    ax2.axvline(x=non_overlap_start_ms, color='gray', linestyle='--', alpha=0.7)
    
    if title:
        plt.suptitle(title)
    plt.xlabel('Time (ms)')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_png) if os.path.dirname(output_png) else '.', exist_ok=True)
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
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


def process_single_report(report, output_base_dir, gpu_type="ampere", force_sqlite=False, skip_timeline=False):
    """Process a single nsys-rep file.
    
    SQLite generation is skipped if file exists (unless force_sqlite=True).
    PNG timeline is always regenerated.
    """
    # Determine output directory structure
    rel_dir = os.path.relpath(report["dirname"], os.path.dirname(report["dirname"]))
    output_dir = os.path.join(output_base_dir, report["frequency"], report["phase"])
    
    # Export to SQLite (skip if exists, unless force_sqlite)
    sqlite_path, csv_path = export_to_sqlite(report["nsys_rep"], output_dir, force=force_sqlite)
    if sqlite_path is None:
        return None
    
    result = {
        "basename": report["basename"],
        "frequency": report["frequency"],
        "phase": report["phase"],
        "overlap_start": report["overlap_start"],
        "overlap_end": report["overlap_end"],
        "sm_num": report["sm_num"],
        "block_size": report["block_size"],
        "sqlite_path": sqlite_path,
        "csv_path": csv_path,
    }
    
    if not skip_timeline:
        # Get kernel data and draw timeline (always regenerate PNG)
        kernel_df = get_kernel_data_from_sqlite(sqlite_path)
        output_png = os.path.join(output_dir, f"{report['basename']}.timeline.png")
        title = f"{report['phase']} | ({report['overlap_start']},{report['overlap_end']}) | SM {report['sm_num']}, Block {report['block_size']} | Freq {report['frequency']}"
        
        ok, stats = draw_timeline_from_kernel_data(
            kernel_df,
            output_png,
            (report["overlap_start"], report["overlap_end"]),
            title=title,
            gpu_type=gpu_type,
            phase=report["phase"],
        )
        if ok:
            print(f"[✓] Timeline: {output_png}")
            result.update(stats)
            result["timeline_png"] = output_png
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Process nsys profile results: export to SQLite and draw timelines")
    parser.add_argument("--input_dir", "-i", type=str, required=True,
                       help="Input directory containing nsys-rep files (e.g., profile_result/tp8-bs8-seq4096)")
    parser.add_argument("--output_dir", "-o", type=str, default="",
                       help="Output directory for SQLite and timelines (default: results/<input_dir_basename>)")
    parser.add_argument("--gpu_type", type=str, default="ampere",
                       help="GPU type: hopper or ampere (affects kernel name matching)")
    parser.add_argument("--force-sqlite", "-f", action="store_true",
                       help="Force re-export SQLite even if it exists (PNG is always regenerated)")
    parser.add_argument("--skip_timeline", action="store_true",
                       help="Skip timeline drawing")
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"[!] Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Derive from input_dir: profile_result/X -> results/X
        input_basename = os.path.basename(os.path.normpath(args.input_dir))
        parent_dir = os.path.dirname(os.path.normpath(args.input_dir))
        if "event" in parent_dir:
            output_dir = os.path.join("results", "event", input_basename)
        else:
            output_dir = os.path.join("results", input_basename)
    
    print(f"[*] Input directory: {args.input_dir}")
    print(f"[*] Output directory: {output_dir}")
    
    # Find all nsys-rep files
    reports = find_nsys_reports(args.input_dir)
    if not reports:
        print(f"[!] No nsys-rep files found in {args.input_dir}")
        sys.exit(1)
    
    print(f"[*] Found {len(reports)} nsys-rep files")
    
    # Process each report
    all_results = []
    for report in reports:
        result = process_single_report(
            report,
            output_dir,
            gpu_type=args.gpu_type,
            force_sqlite=args.force_sqlite,
            skip_timeline=args.skip_timeline,
        )
        if result:
            all_results.append(result)
    
    # Save summary CSV
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_csv = os.path.join(output_dir, "processing_summary.csv")
        os.makedirs(output_dir, exist_ok=True)
        summary_df.to_csv(summary_csv, index=False)
        print(f"[✓] Summary written: {summary_csv}")
        print(f"[✓] Processed {len(all_results)} reports")
    else:
        print("[!] No results to save")


if __name__ == "__main__":
    main()

