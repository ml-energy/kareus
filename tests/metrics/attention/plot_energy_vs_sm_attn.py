import os
import argparse
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

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
mpl.rcParams["font.size"] = 16
mpl.rcParams["axes.titlesize"] = 16
mpl.rcParams["axes.labelsize"] = 16
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["legend.fontsize"] = 16
mpl.rcParams["legend.title_fontsize"] = 16
mpl.rcParams["figure.titlesize"] = 18


def find_energy_columns(column_names: List[str]) -> List[str]:
    suffix = ":total energy (J)"
    energy_columns = [c for c in column_names if c.endswith(suffix)]
    return energy_columns


def read_energy_results(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"energy_results.csv not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = [
        "overlap_start",
        "overlap_end",
        "comm_sm_number",
        "comm_block_size",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"energy_results.csv missing columns: {missing}")

    energy_columns = find_energy_columns(list(df.columns))
    if not energy_columns:
        # Fallback to a common single column naming if present
        if "0:total energy (J)" in df.columns:
            energy_columns = ["0:total energy (J)"]
        else:
            raise ValueError("No energy columns found matching '*:total energy (J)'")

    df["mean_total_energy_J"] = df[energy_columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    # Normalize dtypes used downstream
    for c in ["overlap_start", "overlap_end", "comm_sm_number", "comm_block_size"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    return df


def _save_figure_all_formats(fig, base_output_path: str, formats: List[str]) -> List[str]:
    outputs: List[str] = []
    base_root, base_ext = os.path.splitext(base_output_path)
    for fmt in formats:
        fmt = fmt.lower().strip().lstrip('.')
        out_path = base_output_path if (base_ext.lower().lstrip('.') == fmt) else f"{base_root}.{fmt}"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        metadata = None
        if fmt == "pdf":
            metadata = {"CreationDate": None}
        elif fmt == "svg":
            metadata = {"Date": None}
        fig.savefig(out_path, bbox_inches="tight", dpi=150, metadata=(metadata or {}))
        outputs.append(out_path)
    return outputs


def _normalize_csv_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = {c.lower(): c for c in df.columns}
    return {
        "kernel": cols.get("name".lower()),
        "device": cols.get("device".lower()) or cols.get("device id".lower()),
        "min_ns": cols.get("min (ns)".lower()),
        "avg_ns": cols.get("avg (ns)".lower()),
        "max_ns": cols.get("max (ns)".lower()),
        "total_ns": cols.get("total (ns)".lower()),
    }


def _get_fc_kernel_names(gpu_type: str) -> Tuple[List[str], List[str]]:
    if (gpu_type or "").lower() == "ampere":
        fc1_names = ["ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn"]
        fc2_names = ["ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn"]
    else:  # hopper (default)
        fc1_names = ["nvjet_tst_192x192_64x4_1x1_h_bz_coopB_TNN"]
        fc2_names = ["nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN"]
    return fc1_names, fc2_names


def _compute_group_durations_attn(df: pd.DataFrame, cols: Dict[str, Optional[str]], gpu_type: str) -> List[Dict[str, object]]:
    col_kernel = cols["kernel"]
    col_min = cols["min_ns"]
    col_avg = cols["avg_ns"]
    col_max = cols["max_ns"]

    # exclude FillFunctor kernels
    if col_kernel:
        df = df[~df[col_kernel].str.contains("FillFunctor", na=False)]

    fc1_names, fc2_names = _get_fc_kernel_names(gpu_type)
    groups = [
        ("pre", (0, 1), ["triton_poi_fused_add_0", "triton_poi_fused_native_dropout_1", "rmsnorm_fwd_general_kernel"]),
        ("fc1", (2, 3), fc1_names),
        ("post", (4, 5), ["at::native::direct_copy_kernel_cuda", "fused_rope_forward"]),
        ("attn", (6, 6), ["flash_fwd_kernel"]),
        ("fc2", (7, 8), fc2_names),
    ]

    def aggregate_duration(names: List[str], mode: str) -> float:
        if not names:
            return 0.0
        mask = False
        for nm in names:
            cur = df[col_kernel].str.contains(nm, na=False) if col_kernel else False
            mask = cur if isinstance(mask, bool) else (mask | cur)
        if isinstance(mask, bool):
            return 0.0
        matched_rows = df.loc[mask]
        if matched_rows.empty:
            return 0.0
        if mode == "fc1_max" and col_max:
            return float(matched_rows[col_max].astype(float).sum())
        if mode == "fc2_min" and col_min:
            return float(matched_rows[col_min].astype(float).sum())
        # Default: sum of Avg (ns)
        return float(matched_rows[col_avg].astype(float).sum()) if col_avg else 0.0

    group_durations = []
    for key, (gstart, gend), names in groups:
        if key == "fc1":
            duration = aggregate_duration(names, "fc1_max")
        elif key == "fc2":
            duration = aggregate_duration(names, "fc2_min")
        else:
            duration = aggregate_duration(names, "sum_avg")
        group_durations.append({
            "key": key,
            "start_idx": gstart,
            "end_idx": gend,
            "names": names,
            "duration": duration,
        })
    return group_durations


def _compute_ar_latency_ns(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> float:
    col_kernel = cols["kernel"]
    col_min = cols["min_ns"]
    if not (col_kernel and col_min):
        return 0.0
    ar_rows = df[df[col_kernel] == "allreduceKernelEntryPointBF16"]
    if ar_rows.empty:
        return 0.0
    return float(ar_rows[col_min].min())


def derive_real_overlap_info_from_timeline(
    csv_path: str,
    nominal: Tuple[int, int],
    gpu_type: str,
) -> Tuple[Tuple[int, int], List[str]]:
    ov_start, ov_end = nominal
    # Handle explicit no-overlap
    if ov_start == -1 and ov_end == -1:
        return (-1, -1), []
    if not os.path.exists(csv_path):
        return (ov_start, ov_end), []

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return (ov_start, ov_end), []
        cols = _normalize_csv_columns(df)
        # Basic required columns
        if not (cols["kernel"] and cols["avg_ns"] and cols["min_ns"]):
            return (ov_start, ov_end), []

        # AllReduce latency
        ar_latency = _compute_ar_latency_ns(df, cols)
        if ar_latency <= 0:
            return (ov_start, ov_end), []

        group_durations = _compute_group_durations_attn(df, cols, gpu_type)

        # Determine groups intersecting the nominal window and order by start index
        overlap_groups = [g for g in group_durations if not (g["end_idx"] < ov_start or g["start_idx"] > ov_end)]
        overlap_groups.sort(key=lambda g: g["start_idx"])

        # Accumulate durations until AR latency is covered
        cum = 0.0
        real_end = ov_end
        overlapped_keys: List[str] = []
        for g in overlap_groups:
            dur = float(g["duration"]) if g["duration"] else 0.0
            if dur > 0:
                overlapped_keys.append(str(g["key"]))
            cum += dur
            if cum >= ar_latency:
                real_end = int(g["end_idx"]) 
                break
        return (ov_start, real_end), overlapped_keys
    except Exception:
        return (ov_start, ov_end), []


def attach_real_overlap_scopes(
    df: pd.DataFrame,
    input_dir: str,
    gpu_type: str,
) -> pd.DataFrame:
    # Compute once per unique configuration
    cache: Dict[Tuple[int, int, int, int], Tuple[int, int]] = {}
    real_starts = []
    real_ends = []
    for _, row in df.iterrows():
        ov_s = int(row["overlap_start"]) if pd.notna(row["overlap_start"]) else -1
        ov_e = int(row["overlap_end"]) if pd.notna(row["overlap_end"]) else -1
        sm = int(row["comm_sm_number"]) if pd.notna(row["comm_sm_number"]) else -1
        bs = int(row["comm_block_size"]) if pd.notna(row["comm_block_size"]) else -1
        key = (ov_s, ov_e, sm, bs)
        if key not in cache:
            base = f"profile_{ov_s}_{ov_e}_{sm}_{bs}"
            csv_path = os.path.join(input_dir, f"{base}.csv_cuda_gpu_kern_sum.csv")
            (real_scope, overlapped_keys) = derive_real_overlap_info_from_timeline(csv_path, (ov_s, ov_e), gpu_type)
            cache[key] = real_scope
            # Print nominal vs real once per config
            print(f"Config (SM={sm}, BS={bs}): nominal=({ov_s},{ov_e}) -> real=({real_scope[0]},{real_scope[1]}) overlapped={[k for k in overlapped_keys]} [{os.path.basename(csv_path)}]")
        r_s, r_e = cache[key]
        real_starts.append(r_s)
        real_ends.append(r_e)

    df = df.copy()
    df["real_overlap_start"] = real_starts
    df["real_overlap_end"] = real_ends
    # Build human-readable overlap label using group names
    # Re-derive labels per row using real start/end
    group_labels = {
        'pre': 'RMSNorm',
        'fc1': 'Linear1',
        'post': 'RoPE',
        'attn': 'Attention',
        'fc2': 'Linear2',
    }
    group_ranges = [
        ("pre", (0, 1)),
        ("fc1", (2, 3)),
        ("post", (4, 5)),
        ("attn", (6, 6)),
        ("fc2", (7, 8)),
    ]

    def overlapped_label_for_scope(rs: int, re: int) -> str:
        if rs == -1 and re == -1:
            return "No overlap"
        overlapped_names: List[str] = []
        for key_name, (s, e) in group_ranges:
            if not (e < rs or s > re):
                overlapped_names.append(group_labels.get(key_name, key_name))
        return "+".join(overlapped_names) if overlapped_names else "No overlap"

    df["real_overlap_label"] = [
        overlapped_label_for_scope(int(rs), int(re))
        for rs, re in zip(df["real_overlap_start"], df["real_overlap_end"])
    ]
    return df


def plot_energy_vs_sm(
    df: pd.DataFrame,
    title: str,
    output_path: str,
    cmap_name: str = "tab20",
    legend_title_override: Optional[str] = None,
) -> None:
    # Legend labels are overlapped group names (not start/end indices)

    # Keep only needed columns and enforce SM filter (<= 20)
    filtered = df[[
        "overlap_start",
        "overlap_end",
        "real_overlap_start" if "real_overlap_start" in df.columns else "overlap_start",
        "real_overlap_end" if "real_overlap_end" in df.columns else "overlap_end",
        "real_overlap_label" if "real_overlap_label" in df.columns else None,
        "legend_start_label" if "legend_start_label" in df.columns else None,
        "comm_sm_number",
        "comm_block_size",
        "mean_total_energy_J",
    ]]
    # Drop potential None column added above
    filtered = filtered.dropna(axis=1, how='all')
    filtered = filtered.dropna(subset=["comm_sm_number", "mean_total_energy_J"])
    filtered = filtered[pd.to_numeric(filtered["comm_sm_number"], errors="coerce") <= 20]
    # Ensure uniqueness: one point per (real overlap scope, SM, block size)
    if "real_overlap_start" in filtered.columns and "real_overlap_end" in filtered.columns:
        dup_key = ["real_overlap_start", "real_overlap_end", "comm_sm_number", "comm_block_size"]
    else:
        dup_key = ["overlap_start", "overlap_end", "comm_sm_number", "comm_block_size"]
    filtered = filtered.drop_duplicates(subset=dup_key, keep='first')

    if filtered.empty:
        raise RuntimeError("No valid rows to plot after filtering.")

    plt.figure(figsize=(10, 6))

    # Color by overlapped group label if available, else by (overlap_start, overlap_end)
    # Helper to assign colors: use Dark2 first, then fall back to tab20
    def build_colors_for_labels(labels_order: List[str]) -> Dict[str, object]:
        def get_listed_colors(name: str) -> List[object]:
            try:
                cm = plt.get_cmap(name)
            except Exception:
                return []
            if hasattr(cm, 'colors') and isinstance(getattr(cm, 'colors'), (list, tuple)) and len(getattr(cm, 'colors')) > 0:
                return list(cm.colors)
            # Not a listed colormap; sample a fixed number
            sample_n = 20
            return [cm((i + 0.5) / sample_n) for i in range(sample_n)]

        primary = get_listed_colors('Dark2')
        secondary = get_listed_colors('tab20')
        palette = list(primary) + secondary[6:8] + secondary[12:13]
        if len(palette) == 0:
            # Absolute fallback
            fallback = plt.get_cmap('tab20')
            palette = [fallback((i + 0.5) / max(len(labels_order), 1)) for i in range(max(len(labels_order), 1))]
        colors: Dict[str, object] = {}
        n = len(palette)
        for i, lbl in enumerate(labels_order):
            if i < n:
                colors[lbl] = palette[i]
            else:
                # If more labels than palette, continue with evenly spaced samples from tab20
                cm = plt.get_cmap('tab20')
                colors[lbl] = cm((i + 0.5) / (len(labels_order)))
        return colors

    ax = plt.gca()
    # Prefer start-label grouping when available
    if "legend_start_label" in filtered.columns:
        label_series = filtered["legend_start_label"].fillna("No overlap").replace({"": "No overlap"})
        filtered = filtered.assign(_label=label_series)
        groups = filtered.groupby("_label", dropna=False)
        # Desired order as heatmap
        desired_order = ["No overlap", "RMSNorm", "Linear1", "RoPE", "Attention", "Linear2"]
        labels = [l for l in desired_order if l in groups.groups]
        labels += [l for l in sorted(groups.groups.keys()) if l not in labels]
        colors = build_colors_for_labels(labels)
        for lbl in labels:
            group = groups.get_group(lbl)
            x_values = group["comm_sm_number"].astype(int)
            y_values = group["mean_total_energy_J"].astype(float)
            order = x_values.argsort()
            x_sorted = x_values.iloc[order]
            y_sorted = y_values.iloc[order]
            ax.plot(x_sorted, y_sorted, marker='o', linestyle='-', alpha=0.9, label=lbl, color=colors[lbl], markersize=4)
        legend_title = legend_title_override or "Overlap Start"
    elif "real_overlap_label" in filtered.columns:
        # Normalize labels and group on a single column to avoid tuple keys
        label_series = filtered["real_overlap_label"].fillna("No overlap").replace({"": "No overlap"})
        filtered = filtered.assign(_label=label_series)
        groups = filtered.groupby("_label", dropna=False)
        # Build a unique color per label; put 'No overlap' first
        labels = sorted(list(groups.groups.keys()))
        if "No overlap" in labels:
            labels = ["No overlap"] + [l for l in labels if l != "No overlap"]
        colors = build_colors_for_labels(labels)
        for lbl in labels:
            group = groups.get_group(lbl)
            x_values = group["comm_sm_number"].astype(int)
            y_values = group["mean_total_energy_J"].astype(float)
            # Sort by x so lines connect monotonically in SM
            order = x_values.argsort()
            x_sorted = x_values.iloc[order]
            y_sorted = y_values.iloc[order]
            ax.plot(x_sorted, y_sorted, marker='o', linestyle='-', alpha=0.9, label=lbl, color=colors[lbl], markersize=4)
        legend_title = legend_title_override or "Overlap Scope"
    else:
        group_cols = ["real_overlap_start", "real_overlap_end"] if ("real_overlap_start" in filtered.columns and "real_overlap_end" in filtered.columns) else ["overlap_start", "overlap_end"]
        groups = filtered.groupby(group_cols, dropna=False)
        # Build label list in deterministic order
        label_items = list(groups.groups.keys())
        labels = [f"({int(s)},{int(e)})" for (s, e) in label_items]
        colors = build_colors_for_labels(labels)
        for ((ovl_start, ovl_end), group), lbl in zip(groups, labels):
            x_values = group["comm_sm_number"].astype(int)
            y_values = group["mean_total_energy_J"].astype(float)
            order = x_values.argsort()
            x_sorted = x_values.iloc[order]
            y_sorted = y_values.iloc[order]
            ax.plot(x_sorted, y_sorted, marker='o', linestyle='-', alpha=0.9, label=lbl, color=colors[lbl], markersize=4)
        legend_title = legend_title_override or ("real overlap" if group_cols[0].startswith("real_") else "overlap")

    plt.xlabel("SM number")
    plt.ylabel("Energy (J)")
    # plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(
        title=legend_title,
        fontsize=mpl.rcParams.get("legend.fontsize", 14),
        title_fontsize=mpl.rcParams.get("legend.title_fontsize", mpl.rcParams.get("legend.fontsize", 14)),
    )
    # Integer ticks for SM axis
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Save figure (PNG + optional SVG/PDF with deterministic metadata)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    fig = plt.gcf()
    _save_figure_all_formats(fig, output_path, formats=["png", "svg", "pdf"])
    plt.close(fig)


def plot_energy_heatmap(
    df: pd.DataFrame,
    title: str,
    output_path: str,
    cmap_name: str = "Greens",
) -> None:
    # Prepare data: require real_overlap_label, comm_sm_number, mean_total_energy_J
    if "real_overlap_label" not in df.columns:
        raise RuntimeError("real_overlap_label column is required to draw heatmap.")

    data_cols = [
        "real_overlap_label",
        "comm_sm_number",
        "mean_total_energy_J",
    ]
    hdf = df[data_cols].dropna(subset=["real_overlap_label", "comm_sm_number", "mean_total_energy_J"]).copy()

    # Unique axes
    sm_values = sorted(pd.to_numeric(hdf["comm_sm_number"], errors="coerce").dropna().astype(int).unique().tolist())
    labels = sorted(hdf["real_overlap_label"].astype(str).unique().tolist())
    # Desired top-to-bottom order
    desired_order = ["No overlap", "RMSNorm", "Linear1", "RoPE", "Attention", "Linear2"]
    labels = [l for l in desired_order if l in labels] + [l for l in labels if l not in desired_order]

    # Map to indices
    sm_to_idx = {sm: i for i, sm in enumerate(sm_values)}
    label_to_idx = {lb: i for i, lb in enumerate(labels)}

    # Build matrix and fill with NaN
    mat = np.full((len(labels), len(sm_values)), np.nan, dtype=float)

    # Deduplicate per (label, sm) and keep first
    hdf = hdf.sort_values(["real_overlap_label", "comm_sm_number"]).drop_duplicates(subset=["real_overlap_label", "comm_sm_number"], keep="first")
    for _, row in hdf.iterrows():
        r = label_to_idx[str(row["real_overlap_label"])]; c = sm_to_idx[int(row["comm_sm_number"])]; mat[r, c] = float(row["mean_total_energy_J"]) 

    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        cmap = plt.get_cmap(cmap_name)
    except Exception:
        cmap = plt.get_cmap("Greens")

    # Display heatmap; align cells with integer SM positions
    # origin='upper' so the first label appears at the top
    # Fix energy color scale to [3, 7] J
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, origin="upper", vmin=3.5, vmax=7)

    # Ticks and labels
    ax.set_xticks(range(len(sm_values)))
    ax.set_xticklabels([str(s) for s in sm_values])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("SM number")
    ax.set_ylabel("Overlap Start")
    # ax.set_title(title)

    # Grid lines to form visible blocks
    ax.set_xticks(np.arange(-0.5, len(sm_values), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1, alpha=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Energy (J)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    _save_figure_all_formats(fig, output_path, formats=["png", "svg", "pdf"])
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot energy vs SM number for Attention logs")
    parser.add_argument("--frequency", "-f", type=str, default="default", help="Frequency tag used in results/logs")
    parser.add_argument("--world_size", "-w", type=int, default=2)
    parser.add_argument("--batch_size", "-b", type=int, default=8)
    parser.add_argument("--seq_len", "-s", type=int, default=4096)
    parser.add_argument("--gpu_type", type=str, default="ampere", help="GPU type: hopper or ampere")
    parser.add_argument("--input_dir", type=str, default="", help="If empty, derives from parameters: profile_result/tp<w>-bs<b>-seq<s>/<f>")
    parser.add_argument("--output", type=str, default=None, help="Optional explicit output PNG path")
    parser.add_argument("--cmap", type=str, default="Dark2", help="Matplotlib colormap name for label colors (e.g., tab20, Set3, tab10, Dark2)")
    parser.add_argument("--heatmap_cmap", type=str, default="RdBu_r", help="Colormap for heatmap (e.g., Greens, viridis)")
    parser.add_argument("--no_heatmap", action="store_true", help="Disable heatmap rendering")
    parser.add_argument("--launch_time_only", action="store_true", help="Consider launch time only: use nominal scopes and restrict to select starts; label by first group name")
    args = parser.parse_args()

    tp_bs_seq = f"tp{args.world_size}-bs{args.batch_size}-seq{args.seq_len}"
    logs_dir = os.path.join("logs", tp_bs_seq, str(args.frequency))
    if args.input_dir and len(args.input_dir.strip()) > 0:
        input_dir = args.input_dir
    else:
        input_dir = os.path.join("profile_result", tp_bs_seq, str(args.frequency))
    energy_csv = os.path.join(logs_dir, "energy_results.csv")

    df = read_energy_results(energy_csv)
    # Only process SM number <= 20
    df = df[pd.to_numeric(df["comm_sm_number"], errors="coerce") <= 20]
    df = df[pd.to_numeric(df["comm_block_size"], errors="coerce") == 1024]

    if args.launch_time_only:
        # Same selection logic idea as heatmap; do not compute real overlap
        ov_s = pd.to_numeric(df["overlap_start"], errors="coerce")
        ov_e = pd.to_numeric(df["overlap_end"], errors="coerce")
        # For attention, allow starts at pre/fc1/post/attn/fc2 starts
        valid_starts = {0, 2, 4, 6, 7}
        mask_valid = ov_s.notna() & ov_e.notna()
        start_int = ov_s.fillna(-999999).astype(int)
        end_int = ov_e.fillna(-999999).astype(int)
        mask_targets = (start_int.isin(valid_starts) & (end_int == 8)) | ((start_int == -1) & (end_int == -1))
        mask = mask_valid & mask_targets
        df = df[mask].copy()
        # Provide start-only legend labels for plotting order
        def start_to_label(v: int) -> str:
            if v == -1:
                return "No overlap"
            if v in (0, 1):
                return "RMSNorm"
            if v in (2, 3):
                return "Linear1"
            if v in (4, 5):
                return "RoPE"
            if v == 6:
                return "Attention"
            if v in (7, 8):
                return "Linear2"
            return f"idx{v}"
        df["legend_start_label"] = start_int.loc[df.index].map(start_to_label)
        # Build human-readable label like heatmap when start=-1 => No overlap, else group names by start..end
        group_labels = {
            'pre': 'RMSNorm',
            'fc1': 'Linear1',
            'post': 'RoPE',
            'attn': 'Attention',
            'fc2': 'Linear2',
        }
        group_ranges = [
            ("pre", (0, 1)),
            ("fc1", (2, 3)),
            ("post", (4, 5)),
            ("attn", (6, 6)),
            ("fc2", (7, 8)),
        ]
        def nominal_label(rs: int, re: int) -> str:
            if rs == -1 and re == -1:
                return "No overlap"
            names: List[str] = []
            for key_name, (s, e) in group_ranges:
                if not (e < rs or s > re):
                    names.append(group_labels.get(key_name, key_name))
            return "+".join(names) if names else "No overlap"
        df["real_overlap_label"] = [nominal_label(int(rs), int(re)) for rs, re in zip(start_int.loc[df.index], end_int.loc[df.index]) ]
    else:
        # Attach real overlap scopes derived from timeline CSVs
        df = attach_real_overlap_scopes(df, input_dir=input_dir, gpu_type=args.gpu_type)

    title = f"Energy vs SM number | {tp_bs_seq} @ {args.frequency}"
    output_path = args.output if args.output else os.path.join(logs_dir, f"attn_energy_vs_sm_{args.frequency}.png")
    legend_title = "Overlap Start" if args.launch_time_only else None
    plot_energy_vs_sm(df, title, output_path, cmap_name=args.cmap, legend_title_override=legend_title)
    print(f"Saved figure to: {output_path}")

    # Heatmap figure
    if not args.no_heatmap:
        heatmap_out = os.path.join(logs_dir, "energy_vs_sm_heatmap_attn.png")
        try:
            plot_energy_heatmap(df, title, heatmap_out, cmap_name=args.heatmap_cmap)
            print(f"Saved heatmap to: {heatmap_out}")
        except Exception as e:
            print(f"[!] Heatmap skipped: {e}")


if __name__ == "__main__":
    main()


