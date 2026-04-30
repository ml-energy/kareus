#!/usr/bin/env python3
"""Build time/energy reduction tables for Perseus, Nanobatch+Perseus,
and Kareus.

For each configuration under ``tests/artifact/nemo_experiments/<model>/<config_tag>/``
we look for the following per-method subdirectories:

    megatron/             (baseline)
    perseus/              -> "Megatron-LM + Perseus"
    nanobatch_perseus/    -> "Nanobatching + Perseus"
    kareus/               -> "Kareus"

Each method directory must contain one ``zeus_monitor_global_rank-*_local_rank-*.txt``
per rank.  For every rank file we keep rows where ``window_name`` equals
``training_step_fwd_bwd_step_call``, drop the first ``--warmup`` iterations
and take the next ``--iters`` rows.

Per configuration/method, reductions are computed per iteration first and then
averaged across the kept iterations:
    time_s   = mean over iterations of (max over ranks of elapsed_time)
    energy_J = mean over iterations of (sum over ranks of gpu0_energy)

Reductions are relative to the ``megatron`` baseline:
    time_red_pct   = 100 * (t_baseline - t_method) / t_baseline
    energy_red_pct = 100 * (e_baseline - e_method) / e_baseline

The final table is rendered as a matplotlib figure (PNG) and also printed to
stdout.  No Pareto plot is drawn -- only a table image is produced.

With ``--frontier``, each method is read from its 10
``frontier/pipeline_*`` run directories.  Perseus is used as the reference
frontier:
    iso-time energy = best method energy with time <= Perseus minimum-time run
    iso-energy time = best method time with energy <= Perseus minimum-energy run
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


WINDOW = "training_step_fwd_bwd_step_call"

METHODS: List[Tuple[str, str]] = [
    ("perseus", "M+P"),
    ("nanobatch_perseus", "N+P"),
    ("kareus", "Kareus"),
]
FRONTIER_METHODS: List[Tuple[str, str]] = [
    ("nanobatch_perseus", "N+P"),
    ("kareus", "Kareus"),
]
BASELINE = "megatron"
FRONTIER_BASELINE = "perseus"

MODEL_DISPLAY: Dict[str, str] = {
    "megatron_llama3.2_3b": "Llama 3.2 3B",
    "megatron_qwen3_1.7b": "Qwen 3 1.7B",
}

CONFIG_RE = re.compile(r"cp(\d+)_tp(\d+)_mbs(\d+)_seq(\d+)")
EPS = 1e-9


def load_point(point_dir: str, warmup: int, iters: int) -> Tuple[float, float]:
    """Return (time_seconds, energy_joules) for a single run directory."""
    rank_files = sorted(
        glob.glob(os.path.join(point_dir, "zeus_monitor_global_rank-*_local_rank-*.txt"))
    )
    if not rank_files:
        raise FileNotFoundError(f"No zeus_monitor files in {point_dir}")

    per_rank_elapsed: List[List[float]] = []
    per_rank_energy: List[List[float]] = []
    for rf in rank_files:
        elapsed: List[float] = []
        energy: List[float] = []
        with open(rf, "r") as f:
            reader = csv.DictReader(f)
            energy_col = next(
                (c for c in (reader.fieldnames or []) if re.fullmatch(r"gpu\d+_energy", c)),
                None,
            )
            if energy_col is None:
                raise ValueError(f"{rf}: no gpuN_energy column in header {reader.fieldnames}")
            for row in reader:
                if row.get("window_name") != WINDOW:
                    continue
                try:
                    elapsed.append(float(row["elapsed_time"]))
                    energy.append(float(row[energy_col]))
                except (KeyError, TypeError, ValueError):
                    continue
        kept_e = elapsed[warmup : warmup + iters]
        kept_en = energy[warmup : warmup + iters]
        if len(kept_e) < iters or len(kept_en) < iters:
            raise ValueError(
                f"{rf}: have {len(elapsed)} elapsed / {len(energy)} energy {WINDOW} rows; "
                f"need >= {warmup + iters}"
            )
        per_rank_elapsed.append(kept_e)
        per_rank_energy.append(kept_en)

    # Reduce across ranks per iteration first: max(time), sum(energy).
    iter_time = [max(rank_vals[i] for rank_vals in per_rank_elapsed) for i in range(iters)]
    iter_energy = [sum(rank_vals[i] for rank_vals in per_rank_energy) for i in range(iters)]

    # Then average across iterations.
    return sum(iter_time) / iters, sum(iter_energy) / iters


def load_frontier_points(
    method_dir: str, warmup: int, iters: int, expected_count: Optional[int]
) -> List[Tuple[str, float, float]]:
    """Return [(plan_tag, time_seconds, energy_joules), ...] for frontier runs."""
    frontier_dir = os.path.join(method_dir, "frontier")
    if not os.path.isdir(frontier_dir):
        raise FileNotFoundError(f"No frontier directory in {method_dir}")

    point_dirs = sorted(
        d for d in glob.glob(os.path.join(frontier_dir, "pipeline_*")) if os.path.isdir(d)
    )
    if expected_count is not None and len(point_dirs) != expected_count:
        raise ValueError(
            f"{frontier_dir}: expected {expected_count} pipeline_* directories, "
            f"found {len(point_dirs)}"
        )
    if not point_dirs:
        raise FileNotFoundError(f"No pipeline_* directories in {frontier_dir}")

    points: List[Tuple[str, float, float]] = []
    errors: List[str] = []
    for point_dir in point_dirs:
        tag = os.path.basename(point_dir)
        try:
            time_s, energy_j = load_point(point_dir, warmup, iters)
        except (FileNotFoundError, ValueError) as err:
            errors.append(f"{tag}: {err}")
            continue
        points.append((tag, time_s, energy_j))

    if errors:
        shown = "; ".join(errors[:3])
        more = f"; ... {len(errors) - 3} more" if len(errors) > 3 else ""
        raise ValueError(f"{frontier_dir}: incomplete frontier points: {shown}{more}")
    if not points:
        raise FileNotFoundError(f"No valid frontier points in {frontier_dir}")
    return points


def parallelism_label(cp: int, tp: int) -> str:
    if cp == 1:
        return f"TP={tp}"
    return f"CP={cp}+TP={tp}"


def discover_configs(root: str) -> List[Tuple[str, str, int, int, int, int]]:
    """Return a sorted list of (model, config_tag, cp, tp, mbs, seq) under root."""
    found: List[Tuple[str, str, int, int, int, int]] = []
    if not os.path.isdir(root):
        return found
    for model in sorted(os.listdir(root)):
        mdir = os.path.join(root, model)
        if not os.path.isdir(mdir):
            continue
        for cfg in sorted(os.listdir(mdir)):
            cdir = os.path.join(mdir, cfg)
            if not os.path.isdir(cdir):
                continue
            m = CONFIG_RE.match(cfg)
            if not m:
                continue
            cp, tp, mbs, seq = (int(x) for x in m.groups())
            found.append((model, cfg, cp, tp, mbs, seq))
    return found


def fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "--"
    return f"{x:+.2f}"


def build_rows(
    root: str, warmup: int, iters: int
) -> List[List[str]]:
    """Build the table body rows (one per configuration)."""
    rows: List[List[str]] = []
    for model, cfg, cp, tp, mbs, seq in discover_configs(root):
        model_disp = MODEL_DISPLAY.get(model, model)
        base_dir = os.path.join(root, model, cfg, BASELINE)
        try:
            t_base, e_base = load_point(base_dir, warmup, iters)
        except (FileNotFoundError, ValueError) as err:
            print(f"[skip] {model}/{cfg}: baseline unavailable ({err})")
            continue

        time_reds: List[Optional[float]] = []
        energy_reds: List[Optional[float]] = []
        for method_key, _ in METHODS:
            mdir = os.path.join(root, model, cfg, method_key)
            try:
                t_m, e_m = load_point(mdir, warmup, iters)
            except (FileNotFoundError, ValueError) as err:
                print(f"[skip] {model}/{cfg}/{method_key}: {err}")
                time_reds.append(None)
                energy_reds.append(None)
                continue
            time_reds.append(100.0 * (t_base - t_m) / t_base)
            energy_reds.append(100.0 * (e_base - e_m) / e_base)

        rows.append(
            [model_disp, parallelism_label(cp, tp), str(mbs), str(seq)]
            + [fmt_pct(v) for v in time_reds]
            + [fmt_pct(v) for v in energy_reds]
        )
    return rows


def best_energy_under_time(
    points: List[Tuple[str, float, float]], max_time: float
) -> Optional[Tuple[str, float, float]]:
    candidates = [p for p in points if p[1] <= max_time + EPS]
    if not candidates:
        return None
    return min(candidates, key=lambda p: (p[2], p[1], p[0]))


def best_time_under_energy(
    points: List[Tuple[str, float, float]], max_energy: float
) -> Optional[Tuple[str, float, float]]:
    candidates = [p for p in points if p[2] <= max_energy + EPS]
    if not candidates:
        return None
    return min(candidates, key=lambda p: (p[1], p[2], p[0]))


def build_frontier_rows(
    root: str, warmup: int, iters: int, expected_count: Optional[int]
) -> List[List[str]]:
    """Build frontier comparison rows using Perseus as the reference frontier."""
    rows: List[List[str]] = []
    for model, cfg, cp, tp, mbs, seq in discover_configs(root):
        model_disp = MODEL_DISPLAY.get(model, model)
        cfg_dir = os.path.join(root, model, cfg)
        try:
            perseus_points = load_frontier_points(
                os.path.join(cfg_dir, FRONTIER_BASELINE), warmup, iters, expected_count
            )
        except (FileNotFoundError, ValueError) as err:
            print(f"[skip] {model}/{cfg}: Perseus frontier unavailable ({err})")
            continue

        perseus_min_time = min(perseus_points, key=lambda p: (p[1], p[2], p[0]))
        perseus_min_energy = min(perseus_points, key=lambda p: (p[2], p[1], p[0]))

        iso_time_energy_reds: List[Optional[float]] = []
        iso_energy_time_reds: List[Optional[float]] = []
        for method_key, _ in FRONTIER_METHODS:
            try:
                method_points = load_frontier_points(
                    os.path.join(cfg_dir, method_key), warmup, iters, expected_count
                )
            except (FileNotFoundError, ValueError) as err:
                print(f"[skip] {model}/{cfg}/{method_key}: {err}")
                iso_time_energy_reds.append(None)
                iso_energy_time_reds.append(None)
                continue

            iso_time_point = best_energy_under_time(method_points, perseus_min_time[1])
            if iso_time_point is None:
                iso_time_energy_reds.append(None)
            else:
                iso_time_energy_reds.append(
                    100.0 * (perseus_min_time[2] - iso_time_point[2]) / perseus_min_time[2]
                )

            iso_energy_point = best_time_under_energy(
                method_points, perseus_min_energy[2]
            )
            if iso_energy_point is None:
                iso_energy_time_reds.append(None)
            else:
                iso_energy_time_reds.append(
                    100.0
                    * (perseus_min_energy[1] - iso_energy_point[1])
                    / perseus_min_energy[1]
                )

        rows.append(
            [model_disp, parallelism_label(cp, tp), str(mbs), str(seq)]
            + [fmt_pct(v) for v in iso_time_energy_reds]
            + [fmt_pct(v) for v in iso_energy_time_reds]
        )
    return rows


def render_table(
    rows: List[List[str]],
    output: str,
    method_labels: Optional[List[str]] = None,
    metric_headers: Optional[List[str]] = None,
    title: Optional[str] = None,
) -> None:
    """Render the rows as a matplotlib table image with grouped headers."""
    if method_labels is None:
        method_labels = [lbl for _, lbl in METHODS]
    if metric_headers is None:
        metric_headers = ["Time Reduction (%)", "Energy Reduction (%)"]
    if title is None:
        title = "Iteration time and energy reductions (%) relative to Megatron-LM"

    top_header = (
        ["Model", "Parallelism", "\u03bcBatch\nSize", "Sequence\nLength"]
        + [metric for metric in metric_headers for _ in method_labels]
    )
    sub_header = ["", "", "", ""] + method_labels * len(metric_headers)

    table_data = [top_header, sub_header] + rows

    ncols = len(top_header)
    nrows_total = len(table_data)
    fig_w = max(10.0, 1.1 * ncols)
    fig_h = 0.55 * nrows_total + 1.0

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()

    tbl = ax.table(cellText=table_data, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.5)

    for (r, c), cell in tbl.get_celld().items():
        if r <= 1:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8eef7")
        if r == 0:
            if c < 4:
                cell.visible_edges = "TBLR"
            elif (c - 4) % len(method_labels) == 0:
                cell.visible_edges = "TBL"
            elif (c - 4) % len(method_labels) == len(method_labels) - 1:
                cell.visible_edges = "TBR"
            else:
                cell.visible_edges = "TB"

    def merge_header(start_col: int, end_col: int, text: str) -> None:
        cells = [tbl[0, c] for c in range(start_col, end_col + 1)]
        for cell in cells[1:]:
            cell.get_text().set_text("")
        total_w = sum(c.get_width() for c in cells)
        cells[0].set_width(total_w)
        cells[0].get_text().set_text(text)
        for cell in cells[1:]:
            cell.set_width(0.0)

    for c in range(4):
        top_cell = tbl[0, c]
        sub_cell = tbl[1, c]
        top_cell.get_text().set_text(sub_header[c] or top_header[c])
        sub_cell.get_text().set_text("")
        sub_cell.set_height(0.0)

    for metric_idx, metric in enumerate(metric_headers):
        start_col = 4 + metric_idx * len(method_labels)
        end_col = start_col + len(method_labels) - 1
        merge_header(start_col, end_col, metric)

    fig.suptitle(
        title,
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved table figure to {output}")


def print_text_table(
    rows: List[List[str]],
    method_labels: Optional[List[str]] = None,
    metric_labels: Optional[List[str]] = None,
) -> None:
    if method_labels is None:
        method_labels = [lbl.replace("\n", " ") for _, lbl in METHODS]
    if metric_labels is None:
        metric_labels = ["dT%", "dE%"]
    headers = (
        ["Model", "Parallelism", "MBS", "Seq"]
        + [f"{metric} ({m})" for metric in metric_labels for m in method_labels]
    )
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for r in rows:
        print(fmt.format(*r))


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--root",
        default=os.path.join(here, "nemo_experiments"),
        help="Root directory holding <model>/<config_tag>/<method>/ subtrees.",
    )
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument(
        "--output",
        default=None,
        help="Output PNG path for the rendered table.",
    )
    p.add_argument(
        "--frontier",
        action="store_true",
        help="Compare method frontier runs against the Perseus frontier.",
    )
    p.add_argument(
        "--frontier-count",
        type=int,
        default=10,
        help="Expected number of frontier pipeline_* directories; <=0 accepts any count.",
    )
    args = p.parse_args()

    output = args.output or os.path.join(
        here,
        "compare_methods_frontier_table.png"
        if args.frontier
        else "compare_methods_table.png",
    )

    if args.frontier:
        frontier_count = args.frontier_count if args.frontier_count > 0 else None
        rows = build_frontier_rows(args.root, args.warmup, args.iters, frontier_count)
    else:
        rows = build_rows(args.root, args.warmup, args.iters)
    if not rows:
        raise SystemExit(f"No complete configurations found under {args.root}.")

    if args.frontier:
        frontier_labels = [lbl for _, lbl in FRONTIER_METHODS]
        print_text_table(
            rows,
            method_labels=frontier_labels,
            metric_labels=["iso-time dE%", "iso-energy dT%"],
        )
        render_table(
            rows,
            output,
            method_labels=frontier_labels,
            metric_headers=[
                "Iso-Time Energy\nReduction (%)",
                "Iso-Energy Time\nReduction (%)",
            ],
            title="Frontier reductions (%) relative to Perseus",
        )
    else:
        print_text_table(rows)
        render_table(rows, output)


if __name__ == "__main__":
    main()
